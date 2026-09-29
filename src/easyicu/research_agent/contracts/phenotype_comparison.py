"""Closed declaration and output gate for post-clustering descriptive comparisons."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from ..authority.declared_levels import closed_planning_levels_for, resolve_typed_levels
from ..canonical_json import canonical_json
from ..concept_availability import require_supported_variable_source
from ..schema import (
    AnalysisStep,
    PhenotypeComparisonSpec,
    ResearchContext,
    TableOneSpec,
    ValidationFinding,
)
from .cohort_product_keys import is_closed_cohort_product_key
from .table_one import table_one_output_findings
from .table_one_semantics import (
    table_one_variable_kind,
    validate_table_one_column_roles,
)

COMPARISON_ACTION = "phenotyping.outcome_by_cluster"
COMPARISON_PRODUCT = "table:outcome_by_cluster"
ASSIGNMENTS_PRODUCT = "table:phenotype_assignments"
#: The signed trajectory suite freezes its classes in the stability owner.
#: Its labels cover only the stays with enough observed windows, and they are
#: described only when that owner's freeze record says they were frozen.
TRAJECTORY_ASSIGNMENTS_PRODUCT = "table:cluster_assignments"
TRAJECTORY_FREEZE_PRODUCT = "artifact:stability_freeze"
TRAJECTORY_FROZEN_STATUS = "candidate_labels_frozen_stability_threshold_passed"
TRAJECTORY_NO_SOLUTION_REASON = "TRAJECTORY_NO_FROZEN_SOLUTION"
#: Each frozen-label source, as the exact typed products it contributes.
FROZEN_LABEL_SOURCES: Mapping[str, tuple[str, ...]] = {
    ASSIGNMENTS_PRODUCT: (ASSIGNMENTS_PRODUCT,),
    TRAJECTORY_ASSIGNMENTS_PRODUCT: (
        TRAJECTORY_ASSIGNMENTS_PRODUCT,
        TRAJECTORY_FREEZE_PRODUCT,
    ),
}
COMPARISON_KIND = "phenotype_comparison"
_GROUP_COLUMN = "__easyicu_frozen_cluster__"


def comparison_spec_sha256(spec: PhenotypeComparisonSpec) -> str:
    return hashlib.sha256(
        canonical_json(spec.model_dump(mode="json")).encode()
    ).hexdigest()


def _typed_comparison_inputs(step: AnalysisStep) -> tuple[str, str]:
    typed = [key for key in step.inputs if ":" in key]
    cohorts = [key for key in typed if is_closed_cohort_product_key(key)]
    if len(cohorts) == 1 and len(typed) == len(set(typed)):
        labels = set(typed) - {cohorts[0]}
        for source, products in FROZEN_LABEL_SOURCES.items():
            if labels == set(products):
                return cohorts[0], source
    raise ValueError("phenotype_comparison_typed_inputs_invalid")


def comparison_cohort_input(step: AnalysisStep) -> str:
    return _typed_comparison_inputs(step)[0]


def comparison_label_source(step: AnalysisStep) -> str:
    """The frozen label product this comparison groups by."""

    return _typed_comparison_inputs(step)[1]


def is_host_comparison_step(step: AnalysisStep) -> bool:
    """Whether this host contract, not an agent-coded owner, owns the step's table."""

    if step.scientific_action_id != COMPARISON_ACTION:
        return False
    try:
        validate_comparison_step(step)
    except ValueError:
        return False
    return True


def validate_comparison_step(
    step: AnalysisStep, context: ResearchContext | None = None
) -> None:
    spec = step.phenotype_comparison_spec
    if spec is None:
        raise ValueError("phenotype_comparison_spec_missing")
    if (
        step.scientific_action_id != COMPARISON_ACTION
        or step.planned_analysis_role != "secondary"
        or step.expected_outputs != [COMPARISON_PRODUCT]
        or step.model_requirements
        or step.table_one_spec is not None
        or step.robustness_replay_spec is not None
        or step.functional_form_spec is not None
    ):
        raise ValueError("phenotype_comparison_step_shape_invalid")
    comparison_cohort_input(step)
    names = {v.name for v in spec.variables}
    if not {spec.identity_column, *names}.issubset(step.inputs):
        raise ValueError("phenotype_comparison_input_mismatch")
    if context is None:
        return
    if spec.identity_column not in context.cohort.id_columns or _GROUP_COLUMN in names:
        raise ValueError("phenotype_comparison_identity_invalid")
    if not names.issubset(v.name for v in context.variables):
        raise ValueError("phenotype_comparison_variable_unknown")
    validate_table_one_column_roles(names, context)
    outcomes = set(context.cohort.outcome_columns) | {context.target_outcome}
    if set(spec.outcome_columns) != outcomes.intersection(names):
        raise ValueError("phenotype_comparison_outcome_invalid")
    descriptors = {v.name: v for v in context.variables}
    for variable in spec.variables:
        descriptor = descriptors[variable.name]
        require_supported_variable_source(descriptor, context.cohort.database)
        levels = closed_planning_levels_for(name=variable.name, variables=descriptors)
        kind = table_one_variable_kind(descriptor, levels)
        if (
            variable.variable_kind != kind
            or (kind == "categorical" and variable.summary != "count_percent")
            or (kind == "continuous" and variable.summary == "count_percent")
        ):
            raise ValueError("phenotype_comparison_summary_incompatible")


def comparison_table_spec(
    spec: PhenotypeComparisonSpec,
    context: ResearchContext,
    groups: list[int],
    *,
    label_source: str = ASSIGNMENTS_PRODUCT,
) -> TableOneSpec:
    """Resolve only host-owned level tokens; the Planner still chooses every row.

    Cross-sectional labels cover every cohort row.  Frozen trajectory labels
    cover only the stays with enough observed windows: the other stays are
    excluded and counted on every row, never described as a class.
    """
    variables = {v.name: v for v in context.variables}
    rows = []
    for variable in spec.variables:
        payload = variable.model_dump(mode="python")
        if variable.levels:
            payload["levels"], _ = resolve_typed_levels(
                name=variable.name, declared=variable.levels, variables=variables
            )
        rows.append(payload)
    return TableOneSpec(
        schema_version="easyicu.table_one/3",
        group_by=_GROUP_COLUMN,
        group_levels=groups,
        variables=rows,
        p_values_required=False,
        p_value_adjustment="not_applicable_data_derived_groups",
        missing_group_policy=(
            "exclude_and_report"
            if label_source == TRAJECTORY_ASSIGNMENTS_PRODUCT
            else "fail_closed"
        ),
    )


def bound_freeze_status(binding: Mapping[str, Any], run_dir: Path) -> str:
    """The stability owner's freeze status, read through its verified binding."""

    root = Path(run_dir).resolve()
    relative = str(binding.get("relative_path") or "")
    path = (root / relative).resolve()
    if (
        not relative
        or root not in path.parents
        or not path.is_file()
        or hashlib.sha256(path.read_bytes()).hexdigest() != binding.get("sha256")
    ):
        raise ValueError("phenotype_comparison_freeze_record_unverified")
    record = json.loads(path.read_text(encoding="utf-8"))
    status = record.get("freeze_status") if isinstance(record, Mapping) else None
    if not isinstance(status, str) or not status:
        raise ValueError("phenotype_comparison_freeze_record_unverified")
    return status


def phenotype_comparison_output_findings(
    *,
    step: AnalysisStep,
    step_summary: Mapping[str, Any],
    context: ResearchContext | None,
    resolved_input_bindings: Mapping[str, Mapping[str, Any]] | None,
    out_dir: Path | None,
) -> list[ValidationFinding]:
    if step.scientific_action_id != COMPARISON_ACTION:
        return []
    try:
        if context is None or out_dir is None or resolved_input_bindings is None:
            raise ValueError("phenotype_comparison_validation_context_missing")
        validate_comparison_step(step, context)
        spec = step.phenotype_comparison_spec
        cohort_key, label_key = _typed_comparison_inputs(step)
        trajectory = label_key == TRAJECTORY_ASSIGNMENTS_PRODUCT
        source_cohort = resolved_input_bindings[cohort_key]
        source_assignments = resolved_input_bindings[label_key]
        expected = {
            "comparison_spec_sha256": comparison_spec_sha256(spec),
            "source_cohort_sha256": source_cohort["sha256"],
            "source_assignments_sha256": source_assignments["sha256"],
        }
        if (
            step_summary.get("step_id") != step.step_id
            or step_summary.get("deterministic_standard_analysis") != COMPARISON_KIND
            or step_summary.get("authority_scope") != "analysis_only"
            or step_summary.get("refit_performed") is not False
            or step_summary.get("paper_authorization_allowed") is not False
            or any(step_summary.get(key) != value for key, value in expected.items())
            or step_summary.get("output_files")
            != {COMPARISON_PRODUCT: "outcome_by_cluster.csv"}
        ):
            raise ValueError("phenotype_comparison_receipt_mismatch")
        path = Path(out_dir) / "outcome_by_cluster.csv"
        if path.is_symlink() or hashlib.sha256(
            path.read_bytes()
        ).hexdigest() != step_summary.get("output_sha256"):
            raise ValueError("phenotype_comparison_output_digest_mismatch")
        if trajectory:
            freeze_binding = resolved_input_bindings[TRAJECTORY_FREEZE_PRODUCT]
            # The step writes under <run>/steps/<step>/outputs.
            freeze_status = bound_freeze_status(freeze_binding, Path(out_dir).parents[2])
            if (
                step_summary.get("assignment_source") != TRAJECTORY_ASSIGNMENTS_PRODUCT
                or step_summary.get("source_freeze_sha256") != freeze_binding["sha256"]
                or step_summary.get("freeze_status") != freeze_status
            ):
                raise ValueError("phenotype_comparison_receipt_mismatch")
            if freeze_status != TRAJECTORY_FROZEN_STATUS:
                # No class was frozen, so none is described: an honest
                # non-solution, like the signed owners' own.
                if (
                    step_summary.get("scientific_status") != "failed_closed"
                    or step_summary.get("reason_code") != TRAJECTORY_NO_SOLUTION_REASON
                    or not pd.read_csv(path).empty
                ):
                    raise ValueError("phenotype_comparison_no_solution_incoherent")
                return []
        counts = step_summary["cluster_counts"]
        groups = sorted(int(key) for key in counts)
        if (
            len(groups) < 2
            or any(str(group) not in counts or group < 0 for group in groups)
            or any(type(value) is not int or value <= 0 for value in counts.values())
        ):
            raise ValueError("phenotype_comparison_cluster_invalid")
        n = sum(counts.values())
        excluded = step_summary.get("n_not_clustered") if trajectory else 0
        if (
            step_summary.get("n_rows") != n
            or source_assignments["product_contract"]["row_count"] != n
            or type(excluded) is not int
            or excluded < 0
            or source_cohort["product_contract"]["row_count"] != n + excluded
        ):
            raise ValueError("phenotype_comparison_membership_mismatch")
        execution_spec = comparison_table_spec(
            spec, context, groups, label_source=label_key
        )
        if step_summary.get("execution_table_spec") != execution_spec.model_dump(
            mode="json"
        ):
            raise ValueError("phenotype_comparison_execution_spec_mismatch")
        table = pd.read_csv(path, dtype={"group": "string"})
        expected_counts = {"Overall": n, **counts}
        if (
            any(
                set(table[key].dropna().astype(str)) != {value}
                or table[key].isna().any()
                for key, value in expected.items()
            )
            or not table.denominator_n.eq(table["group"].map(expected_counts)).all()
            or not table.group_missing_excluded_n.eq(excluded).all()
        ):
            raise ValueError("phenotype_comparison_output_provenance_mismatch")
        expected_roles = {
            v.name: (
                "outcome" if v.name in spec.outcome_columns else "clinical_profile"
            )
            for v in spec.variables
        }
        if not table.variable_role.eq(table.variable.map(expected_roles)).all():
            raise ValueError("phenotype_comparison_output_role_mismatch")
        # Reuse the same missingness, category, summary and SMD gate as Table 1.
        projected = step.model_copy(update={"table_one_spec": execution_spec})
        findings = table_one_output_findings(
            step=projected, out_dir=out_dir, filename="outcome_by_cluster.csv"
        )
        if hashlib.sha256(path.read_bytes()).hexdigest() != step_summary.get(
            "output_sha256"
        ):
            raise ValueError("phenotype_comparison_output_digest_mismatch")
        return findings
    except (ValueError, KeyError, TypeError, OSError, AttributeError) as exc:
        reason = str(exc).partition(":")[0]
        if not reason.startswith("phenotype_comparison_"):
            reason = "phenotype_comparison_output_invalid"
        return [
            ValidationFinding(
                validator="phenotype_comparison_contract",
                severity="error",
                message=str(exc),
                detail={
                    "step_id": step.step_id,
                    "reason": reason,
                    "lower_layer_error_type": type(exc).__name__,
                },
            )
        ]
