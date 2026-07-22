"""Single deny-by-default projection for content leaving the host."""

from __future__ import annotations

import ast
import hashlib
import json
from typing import Any, Iterable, Mapping, Optional, Sequence

from ..authority.table_one_binding import (
    table_one_code_token_value_map,
    table_one_private_code_label_map,
)
from ..schema import AnalysisStep, ResearchContext
from .prompt_variables import (
    compact_fixed_window_trajectory_prompt,
    project_observed_domain,
)


def _compact(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): compacted
            for key, item in value.items()
            if (compacted := _compact(item)) not in (None, "", [], {}, ())
        }
    if isinstance(value, (list, tuple)):
        return [
            item
            for value in value
            if (item := _compact(value)) not in (None, "", [], {}, ())
        ]
    return value


_OUTCOME_SOURCE_LABELS = {
    "icu_mortality": "ICU mortality",
    "hospital_mortality": "hospital mortality",
    "mortality_28d": "28-day mortality",
    "mortality_30d": "30-day mortality",
    "time_to_event_endpoint": "a time-to-event endpoint",
    "length_of_stay": "a length-of-stay outcome",
    "readmission": "a readmission outcome",
}


def _outcome_semantics(variable: Any) -> dict[str, Any]:
    """Render host-bound outcome semantics without forwarding caveat prose."""

    source_concept = str(variable.source_concept or "").strip()
    label = _OUTCOME_SOURCE_LABELS.get(source_concept)
    return _compact(
        {
            "definition": variable.description,
            "source_concept": source_concept or None,
            "host_binding": (
                f"For this analysis, '{variable.name}' is explicitly treated as "
                f"{label} because that is what the typed outcome binding declares."
                if label
                else None
            ),
        }
    )


def outbound_safe_context_payload(
    context: ResearchContext,
    *,
    variable_names: Optional[Iterable[str]] = None,
) -> dict[str, Any]:
    """Return the only patient-derived context shape allowed out of process."""

    selected = (
        None
        if variable_names is None
        else {str(name).lower() for name in variable_names}
    )
    variables: list[dict[str, Any]] = []
    for variable in context.variables:
        if selected is not None and variable.name.lower() not in selected:
            continue
        projected_domain = project_observed_domain(variable.observed_domain)
        if variable.observed_domain is not None:
            for flag in ("is_binary", "is_constant"):
                if isinstance(variable.observed_domain.get(flag), bool):
                    projected_domain[flag] = variable.observed_domain[flag]
        role = variable.role.value
        variables.append(
            _compact(
                {
                    "name": variable.name,
                    "role": role if role != "meta" else None,
                    "dtype": variable.dtype,
                    "unit": variable.unit,
                    "plausibility_range": variable.valid_range,
                    "range_policy": (
                        "flag_only" if variable.valid_range is not None else None
                    ),
                    "out_of_range_action": (
                        "retain_and_flag" if variable.valid_range is not None else None
                    ),
                    "source_concept": (
                        variable.source_concept if role != "meta" else None
                    ),
                    "outcome_semantics": (
                        _outcome_semantics(variable) if role == "outcome" else None
                    ),
                    "is_ordinal": True if variable.is_ordinal else None,
                    "ordinal_cardinality": (
                        len(variable.ordinal_levels)
                        if variable.ordinal_levels
                        else None
                    ),
                    "allowed_aggregations": (
                        [item.value for item in variable.allowed_aggregations]
                        if [item.value for item in variable.allowed_aggregations]
                        != ["any"]
                        else None
                    ),
                    "aggregation_default": (
                        variable.aggregation_default.value
                        if variable.aggregation_default is not None
                        else None
                    ),
                    "observed_shape": projected_domain,
                    "analysis_window": variable.analysis_window,
                    "temporal_resolution": variable.temporal_resolution,
                    "fixed_window_trajectory": (
                        variable.fixed_window_trajectory.model_dump(mode="json")
                        if variable.fixed_window_trajectory is not None
                        else None
                    ),
                    "missingness_semantics": variable.missingness_semantics,
                    "forbidden_transformations": variable.forbidden_transformations,
                    "missingness": (
                        {
                            "fraction_missing": variable.missingness.fraction_missing,
                            "severity": variable.missingness.missingness_severity,
                        }
                        if variable.missingness is not None
                        else None
                    ),
                }
            )
        )
    return _compact(
        {
            "schema": "easyicu.outbound_safe_context/1",
            "research_question": context.research_question,
            "cohort": {
                "cohort_name": context.cohort.cohort_name,
                "database": context.cohort.database,
                "n_stays": context.cohort.n_stays,
                "n_patients": context.cohort.n_patients,
                "inclusion_contract": context.cohort.inclusion_criteria,
                "exclusion_contract": context.cohort.exclusion_criteria,
                "id_columns": context.cohort.id_columns,
                "time_columns": context.cohort.time_columns,
                "outcome_columns": context.cohort.outcome_columns,
            },
            "primary_exposure": context.primary_exposure,
            "target_outcome": context.target_outcome,
            "time_windows": [
                {
                    "name": window.name,
                    "start_hours": window.start_hours,
                    "end_hours": window.end_hours,
                    "anchor": window.anchor,
                }
                for window in context.time_windows
            ],
            "temporal_constraints": [
                {
                    "relation": constraint.relation,
                    "anchor_event": constraint.anchor_event,
                    "target_concept": constraint.target_concept,
                    "start_hours": constraint.start_hours,
                    "end_hours": constraint.end_hours,
                    "aggregation_hint": constraint.aggregation_hint,
                    "executable_repr": constraint.executable_repr,
                }
                for constraint in context.temporal_constraints
            ],
            "cross_database_validation": context.cross_database_validation,
            "explicit_user_choices": (
                context.user_preferences.model_dump(
                    mode="json",
                    exclude_none=True,
                    exclude={"extra_notes"},
                )
                if context.user_preferences is not None
                else None
            ),
            "variables": variables,
        }
    )


def format_outbound_safe_context(
    context: ResearchContext,
    *,
    variable_names: Optional[Iterable[str]] = None,
) -> str:
    payload = outbound_safe_context_payload(context, variable_names=variable_names)
    selected = (
        None
        if variable_names is None
        else {str(name).lower() for name in variable_names}
    )
    trajectory_variables = [
        variable
        for variable in context.variables
        if variable.fixed_window_trajectory is not None
        and (selected is None or variable.name.lower() in selected)
    ]
    compact_projection = compact_fixed_window_trajectory_prompt(trajectory_variables)
    if compact_projection.shared_lines:
        trajectory_names = {variable.name for variable in trajectory_variables}
        payload["variables"] = [
            row
            for row in payload.get("variables", [])
            if row.get("name") not in trajectory_names
        ]
        payload[
            "Shared fixed-window trajectory policies (binding for the member columns below)"
        ] = list(compact_projection.shared_lines)
        payload["fixed_window_trajectory_columns"] = list(
            compact_projection.variable_lines
        )
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


_SAFE_RECORD_KEYS = frozenset(
    {
        "step_id",
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
    }
)


def project_outbound_records(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Project completed records without summaries, labels, logs, or prose."""

    return [
        {
            **{key: record[key] for key in _SAFE_RECORD_KEYS if key in record},
            **(
                {"step_summary": project_outbound_step_summary(record["step_summary"])}
                if isinstance(record.get("step_summary"), Mapping)
                else {}
            ),
        }
        for record in records
    ]


def project_outbound_probe(value: Mapping[str, Any]) -> dict[str, Any]:
    """Keep only host-defined scalar status/count/digest coordinates."""

    allowed: dict[str, Any] = {}
    for key, item in value.items():
        lowered = str(key).lower()
        if lowered in {"status", "step_id", "error_code"} or lowered.endswith(
            ("_count", "_sha256", "_digest", "_id")
        ):
            if isinstance(item, (str, int, float, bool)) or item is None:
                allowed[str(key)] = item
    return allowed


def outbound_safe_script(step: Optional[AnalysisStep], script_text: str) -> str:
    """Replace host-only Table 1 literals in the audit-only script copy."""

    if step is None:
        return script_text
    mapping = table_one_private_code_label_map(step)
    if not mapping:
        return script_text
    try:
        tree = ast.parse(script_text)
    except SyntaxError as exc:
        raise ValueError(
            "cannot safely project Table 1 script for outbound audit"
        ) from exc

    class _Redactor(ast.NodeTransformer):
        def visit_Constant(self, node: ast.Constant) -> ast.AST:  # noqa: N802
            token = mapping.get((type(node.value).__name__, repr(node.value)))
            if token is None:
                return node
            return ast.copy_location(ast.Constant(value=token), node)

    return ast.unparse(_Redactor().visit(tree)) + "\n"


def restore_outbound_safe_script(step: Optional[AnalysisStep], script_text: str) -> str:
    """Restore host-only Table 1 values after an external repair response."""

    if step is None:
        return script_text
    mapping = table_one_code_token_value_map(step)
    if not mapping:
        return script_text
    try:
        tree = ast.parse(script_text)
    except SyntaxError as exc:
        raise ValueError("cannot restore Table 1 labels from repaired code") from exc

    class _Restorer(ast.NodeTransformer):
        def visit_Constant(self, node: ast.Constant) -> ast.AST:  # noqa: N802
            if isinstance(node.value, str) and node.value in mapping:
                return ast.copy_location(ast.Constant(value=mapping[node.value]), node)
            return node

    return ast.unparse(_Restorer().visit(tree)) + "\n"


_SAFE_TEXT_KEYS = frozenset(
    {
        "status",
        "state",
        "severity",
        "validator",
        "error_code",
        "error_type",
        "issue_code",
        "schema",
        "schema_version",
        "sha256",
        "digest",
        "evidence_id",
        "artifact_id",
        "step_id",
        "role",
        "method",
        "semantics_family",
    }
)
_SAFE_OPAQUE_CATEGORY_KEYS = frozenset(
    {
        "category",
        "comparison",
        "group",
        "stratum",
    }
)
_SAFE_AGGREGATE_NUMERIC_KEYS = frozenset(
    {
        "call_count",
        "code_repair_attempts",
        "completion_tokens",
        "concept_audit_error_count",
        "concept_repair_attempts",
        "count",
        "effect_estimate",
        "error_count",
        "estimate",
        "event_count",
        "finding_count",
        "incidence",
        "lower_ci",
        "missing_count",
        "missing_fraction",
        "mortality_fraction",
        "n",
        "n_patients",
        "n_stays",
        "n_total",
        "p_value",
        "prevalence",
        "prompt_tokens",
        "returncode",
        "step_count",
        "total_tokens",
        "upper_ci",
        "warning_count",
    }
)
_SAFE_AGGREGATE_BOOL_KEYS = frozenset(
    {
        "artifact_valid",
        "deterministic_code_fallback",
        "isolation_degraded",
        "paper_authorized",
        "scientific_requirement_complete",
        "timed_out",
    }
)
_SAFE_SUMMARY_MAPPING_KEYS = frozenset({"counts", "metrics"})


def _opaque_scalar(_value: str) -> str:
    # Summaries do not need a reversible label.  A constant token avoids
    # exposing a dictionary-checkable digest of a private category literal.
    return "__easyicu_category__"


def _safe_summary_value(value: Any, *, key: str = "") -> Any:
    lowered = str(key).lower()
    if isinstance(value, Mapping):
        if lowered not in _SAFE_SUMMARY_MAPPING_KEYS:
            return None
        return _compact(
            {
                str(child_key): _safe_summary_value(child, key=str(child_key))
                for child_key, child in value.items()
            }
        )
    if isinstance(value, (list, tuple)):
        return None
    if isinstance(value, str):
        if lowered in _SAFE_TEXT_KEYS:
            return value
        if lowered in _SAFE_OPAQUE_CATEGORY_KEYS:
            return _opaque_scalar(value)
        return None
    if isinstance(value, bool):
        return value if lowered in _SAFE_AGGREGATE_BOOL_KEYS else None
    if isinstance(value, (int, float)):
        return value if lowered in _SAFE_AGGREGATE_NUMERIC_KEYS else None
    if value is None and (
        lowered in _SAFE_TEXT_KEYS
        or lowered in _SAFE_AGGREGATE_NUMERIC_KEYS
        or lowered in _SAFE_AGGREGATE_BOOL_KEYS
    ):
        return value
    return None


def project_outbound_step_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Project only the registered host-owned aggregate-summary schema."""

    return _compact(
        {
            str(key): _safe_summary_value(value, key=str(key))
            for key, value in summary.items()
        }
    )


def _project_process_manifest(parsed: Mapping[str, Any]) -> dict[str, Any]:
    """Project the registered Tier-2 process-manifest schema only."""

    projected = project_outbound_step_summary(parsed)
    errors = parsed.get("errors")
    if isinstance(errors, list):
        safe_errors: list[dict[str, Any]] = []
        for item in errors:
            if not isinstance(item, Mapping):
                continue
            row = {
                key: item[key]
                for key in ("error_code", "severity", "validator")
                if isinstance(item.get(key), str)
            }
            if row:
                safe_errors.append(row)
        if safe_errors:
            projected["errors"] = safe_errors
    evidence = parsed.get("evidence")
    if isinstance(evidence, Mapping):
        safe_evidence = {
            key: evidence[key]
            for key in ("count", "sha256", "digest")
            if (
                key in evidence
                and (
                    (key == "count" and isinstance(evidence[key], (int, float)))
                    or (key != "count" and isinstance(evidence[key], str))
                )
            )
        }
        if safe_evidence:
            projected["evidence"] = safe_evidence
    return projected


def project_outbound_artifact_bundle(bundle: Mapping[str, str]) -> dict[str, Any]:
    """Expose identity plus safe host-generated structured process evidence."""

    projected: dict[str, Any] = {}
    for name, content in sorted(bundle.items()):
        if str(name).startswith("__"):
            continue
        row: dict[str, Any] = {
            "sha256": hashlib.sha256(str(content).encode("utf-8")).hexdigest(),
            "chars": len(str(content)),
        }
        if str(name) in {"manifest.json", "run_manifest.json", "run_status.json"}:
            try:
                parsed = json.loads(str(content))
            except (TypeError, ValueError):
                parsed = None
            if isinstance(parsed, Mapping):
                structured = _project_process_manifest(parsed)
                if structured:
                    row["structured"] = structured
        projected[str(name)] = row
    return projected


__all__ = [
    "format_outbound_safe_context",
    "outbound_safe_context_payload",
    "outbound_safe_script",
    "project_outbound_artifact_bundle",
    "project_outbound_probe",
    "project_outbound_records",
    "project_outbound_step_summary",
    "restore_outbound_safe_script",
]
