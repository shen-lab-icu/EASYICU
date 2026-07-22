"""Single deny-by-default projection for content leaving the host."""

from __future__ import annotations

import ast
import hashlib
import json
from typing import Any, Iterable, Mapping, Optional, Sequence

from ..authority.table_one_binding import table_one_private_label_map
from ..schema import AnalysisStep, ResearchContext
from .prompt_variables import project_observed_domain


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
        role = variable.role.value
        variables.append(
            _compact(
                {
                    "name": variable.name,
                    "role": role if role != "meta" else None,
                    "dtype": variable.dtype,
                    "source_concept": (
                        variable.source_concept if role != "meta" else None
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
                "database": context.cohort.database,
                "n_stays": context.cohort.n_stays,
                "n_patients": context.cohort.n_patients,
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
            "variables": variables,
        }
    )


def format_outbound_safe_context(
    context: ResearchContext,
    *,
    variable_names: Optional[Iterable[str]] = None,
) -> str:
    return json.dumps(
        outbound_safe_context_payload(context, variable_names=variable_names),
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
    records: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Project completed records without summaries, labels, logs, or prose."""

    return [
        {key: record[key] for key in _SAFE_RECORD_KEYS if key in record}
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
    mapping = table_one_private_label_map(step)
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


def project_outbound_artifact_bundle(bundle: Mapping[str, str]) -> dict[str, Any]:
    """Expose artifact identity and size, never arbitrary artifact text."""

    return {
        str(name): {
            "sha256": hashlib.sha256(str(content).encode("utf-8")).hexdigest(),
            "chars": len(str(content)),
        }
        for name, content in sorted(bundle.items())
        if not str(name).startswith("__")
    }


__all__ = [
    "format_outbound_safe_context",
    "outbound_safe_context_payload",
    "outbound_safe_script",
    "project_outbound_artifact_bundle",
    "project_outbound_probe",
    "project_outbound_records",
]
