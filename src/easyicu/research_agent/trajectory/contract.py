"""Generic contracts for agent-owned fixed-window trajectory phenotyping.

This module never chooses a feature family, cohort threshold, clustering
method, or number of clusters.  It parses representation metadata declared by
the data and verifies only facts that can be replayed from the locked cohort
and agent-produced artifacts.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import pandas as pd
from pydantic import ValidationError

from ..schema import (
    AnalysisStep,
    ClusterSelectionManifest,
    ConceptDescriptor,
    FixedWindowTrajectoryMetadata,
    ResearchContext,
    ValidationFinding,
)

_FIXED_WINDOW_COLUMN = re.compile(
    r"^(?P<family>[A-Za-z][A-Za-z0-9_]*)_h"
    r"(?P<start>\d+(?:p\d+)?)_(?P<end>\d+(?:p\d+)?)$"
)
TRAJECTORY_PHENOTYPING_REQUIRED_OUTPUTS = (
    "manifest:trajectory_missingness_policy",
    "table:trajectory_membership",
    "table:cluster_assignments",
    "table:trajectory_profiles",
    "table:cohort_flow",
    "table:cluster_sizes",
    "table:cluster_stability",
    "table:cluster_stability_assignments",
    "manifest:cluster_selection",
)

_TRAJECTORY_ARTIFACTS = {
    "policy": "trajectory_missingness_policy.json",
    "membership": "trajectory_membership.csv",
    "assignments": "cluster_assignments.csv",
    "profiles": "trajectory_profiles.csv",
    "flow": "cohort_flow.csv",
    "sizes": "cluster_sizes.csv",
    "stability": "cluster_stability.csv",
    "stability_assignments": "cluster_stability_assignments.csv",
    "selection": "cluster_selection.json",
}

_OUTCOME_ARTIFACT = "outcome_by_cluster.csv"
_STRUCTURED_OUTCOME_PRODUCTS = frozenset(
    {
        "outcome_by_cluster",
        "cluster_outcomes",
        "cluster_outcome_summary",
        "cluster_mortality",
    }
)

_TRAJECTORY_PHENOTYPING_METHODS = frozenset(
    {
        "trajectory_clustering",
        "trajectory_clustering_analysis",
        "trajectory_feature_clustering",
        "clustering",
        "kmeans",
        "k_means",
        "kmeans_clustering",
        "k_means_clustering",
        "phenotyping",
        "phenotype_clustering",
        "unsupervised_clustering",
        "latent_class",
        "latent_class_analysis",
        "latent_class_model",
        "cluster_analysis",
        "gmm",
        "gaussian_mixture",
        "gaussian_mixture_model",
    }
)


def _window_number(value: str) -> float:
    return float(str(value).replace("p", "."))


def infer_fixed_window_trajectory_metadata(
    *,
    column_name: str,
    values: pd.Series,
    source_scale: str,
) -> Optional[FixedWindowTrajectoryMetadata]:
    """Parse ``<family>_h<start>_<end>`` and classify its representation.

    Fractional observed values are evidence that a discrete source scale has
    already been summarized within each fixed window.  That representation is
    distinct from a raw integer ordinal state; the source scale remains
    recorded so downstream reporting does not lose the original semantics.
    """

    match = _FIXED_WINDOW_COLUMN.fullmatch(str(column_name or ""))
    if match is None:
        return None
    start = _window_number(match.group("start"))
    end = _window_number(match.group("end"))
    if end <= start:
        return None

    numeric = pd.to_numeric(values, errors="coerce").dropna()
    observed_fractional = bool(
        not numeric.empty
        and ((numeric.astype(float) - numeric.astype(float).round()).abs() > 1e-8).any()
    )
    normalized_scale = str(source_scale or "unknown").strip().lower()
    allowed_scales = {
        "continuous",
        "ordinal",
        "binary",
        "categorical",
        "count",
        "unknown",
    }
    if normalized_scale not in allowed_scales:
        normalized_scale = "unknown"

    if observed_fractional:
        representation_kind = "fractional_window_summary"
    elif normalized_scale == "continuous":
        representation_kind = "continuous_window_summary"
    elif normalized_scale in {"ordinal", "binary", "categorical", "count"}:
        representation_kind = "discrete_window_state"
    else:
        representation_kind = "unknown_window_representation"

    return FixedWindowTrajectoryMetadata(
        family=match.group("family"),
        window_start_hours=start,
        window_end_hours=end,
        window_width_hours=end - start,
        source_scale=normalized_scale,  # type: ignore[arg-type]
        representation_kind=representation_kind,
        observed_fractional_values=observed_fractional,
    )


def is_continuous_trajectory_representation(variable: ConceptDescriptor) -> bool:
    metadata = variable.fixed_window_trajectory
    return bool(
        metadata is not None
        and metadata.representation_kind
        in {"fractional_window_summary", "continuous_window_summary"}
    )


def _regex_selects_column(
    pattern: str, column: str, *, fullmatch: bool = False
) -> bool:
    try:
        compiled = re.compile(pattern)
    except re.error:
        return False
    return bool(compiled.fullmatch(column) if fullmatch else compiled.search(column))


def _selector_mentions_trajectory(node: ast.AST, trajectory_columns: set[str]) -> bool:
    """Recognize common explicit/dynamic DataFrame column selectors."""

    for child in ast.walk(node):
        if isinstance(child, ast.Constant) and isinstance(child.value, str):
            if child.value in trajectory_columns:
                return True
        if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute):
            operation = child.func.attr
            if operation in {"startswith", "endswith"} and child.args:
                token_node = child.args[0]
                if isinstance(token_node, ast.Constant) and isinstance(
                    token_node.value, str
                ):
                    token = token_node.value
                    if any(
                        (
                            column.startswith(token)
                            if operation == "startswith"
                            else column.endswith(token)
                        )
                        for column in trajectory_columns
                    ):
                        return True
            if operation == "filter":
                for keyword in child.keywords:
                    if not isinstance(keyword.value, ast.Constant) or not isinstance(
                        keyword.value.value, str
                    ):
                        continue
                    token = keyword.value.value
                    if keyword.arg == "like" and any(
                        token in column for column in trajectory_columns
                    ):
                        return True
                    if keyword.arg == "regex" and any(
                        _regex_selects_column(token, column)
                        for column in trajectory_columns
                    ):
                        return True
        if isinstance(child, ast.Call):
            operation = (
                child.func.attr
                if isinstance(child.func, ast.Attribute)
                else child.func.id if isinstance(child.func, ast.Name) else ""
            )
            if operation in {"match", "search", "fullmatch"} and child.args:
                pattern_node = child.args[0]
                if isinstance(pattern_node, ast.Constant) and isinstance(
                    pattern_node.value, str
                ):
                    if any(
                        _regex_selects_column(
                            pattern_node.value,
                            column,
                            fullmatch=operation == "fullmatch",
                        )
                        for column in trajectory_columns
                    ):
                        return True
        if isinstance(child, ast.Compare) and any(
            isinstance(operation, ast.In) for operation in child.ops
        ):
            if isinstance(child.left, ast.Constant) and isinstance(
                child.left.value, str
            ):
                if any(child.left.value in column for column in trajectory_columns):
                    return True
    return False


def selected_trajectory_variables(
    *,
    context: ResearchContext,
    script_text: str,
    step: Optional[AnalysisStep] = None,
) -> list[ConceptDescriptor]:
    """Return trajectory columns selected literally, by prefix, or by plan.

    Plan ``inputs`` are semantic declarations and therefore prevent a code
    spelling change (literal list versus ``startswith``) from changing the
    compatibility decision.
    """

    trajectory = [
        variable
        for variable in context.variables
        if variable.fixed_window_trajectory is not None
    ]
    if not trajectory:
        return []
    code = str(script_text or "")
    input_names = {str(value) for value in (step.inputs if step is not None else [])}
    try:
        tree: Optional[ast.AST] = ast.parse(code)
    except SyntaxError:
        tree = None
    selected: list[ConceptDescriptor] = []
    for variable in trajectory:
        literal = (
            re.search(
                rf"(?<![A-Za-z0-9_]){re.escape(variable.name)}(?![A-Za-z0-9_])",
                code,
            )
            is not None
        )
        dynamically_selected = bool(
            tree is not None and _selector_mentions_trajectory(tree, {variable.name})
        )
        if variable.name in input_names or literal or dynamically_selected:
            selected.append(variable)
    return selected


def _method_head(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")
    return normalized.split("_with_", 1)[0]


def trajectory_phenotyping_contract_applies(
    *, context: ResearchContext, step: AnalysisStep
) -> bool:
    if _method_head(step.method or "") not in _TRAJECTORY_PHENOTYPING_METHODS:
        return False
    input_names = {str(value) for value in (step.inputs or [])}
    bins_by_family: dict[str, list[FixedWindowTrajectoryMetadata]] = {}
    for variable in context.variables:
        metadata = variable.fixed_window_trajectory
        if variable.name not in input_names or metadata is None:
            continue
        bins_by_family.setdefault(metadata.family, []).append(metadata)
    for bins in bins_by_family.values():
        ordered = sorted(
            bins,
            key=lambda item: (item.window_start_hours, item.window_end_hours),
        )
        if len(ordered) < 2:
            continue
        if len(
            {(item.window_start_hours, item.window_end_hours) for item in ordered}
        ) != len(ordered):
            continue
        if all(
            left.window_end_hours <= right.window_start_hours
            for left, right in zip(ordered, ordered[1:])
        ):
            return True
    return False


def trajectory_outcome_contract_declared(step: AnalysisStep) -> bool:
    """Return whether the agent explicitly planned a structured outcome product."""

    for raw in step.expected_outputs or []:
        kind, separator, product = str(raw or "").strip().lower().partition(":")
        if not separator or kind not in {"table", "dataset", "artifact", "manifest"}:
            continue
        normalized = re.sub(r"\.(?:csv|json|parquet)$", "", product.rsplit("/", 1)[-1])
        if normalized in _STRUCTURED_OUTCOME_PRODUCTS:
            return True
    return False


def trajectory_phenotyping_code_contract(
    *, context: ResearchContext, step: AnalysisStep
) -> str:
    """Coder-facing schema for replayable, agent-owned trajectory products."""

    if not trajectory_phenotyping_contract_applies(context=context, step=step):
        return ""
    outcome_contract = (
        "- Because this plan explicitly declares an outcome-by-cluster product, "
        "declare outcome_summary_statistic ('mean' or 'median') in the policy "
        "when the outcome is non-binary, and "
        "write outcome_by_cluster.csv. For a binary outcome write "
        "cluster,n,outcome_n,event_n,outcome_rate; for a non-binary outcome "
        "write cluster,n,outcome_n,summary_statistic,value.\n"
        if trajectory_outcome_contract_declared(step)
        else ""
    )
    return (
        (
            "\n\nFIXED-WINDOW TRAJECTORY PHENOTYPING CONTRACT (method, k, family, "
            "threshold, and summaries remain your choices):\n"
            "- Write trajectory_missingness_policy.json with id_column, "
            "observation_family, observation_columns, min_observed_windows, "
            "profile_columns, profile_summary_statistic ('mean' or 'median'), "
            "clustering_method, n_clusters, time_axis='relative_hours', and "
            "anchor, anchor_provenance, anchor_source. When ResearchContext has "
            "one explicit relative-to-anchor task constraint, copy it with "
            "anchor_provenance='task_contract'; otherwise explicitly use "
            "anchor_provenance='agent_declared' and name the planning/data source. "
            "For task-contract provenance, set "
            "anchor_source='temporal_constraints.relative_to_anchor'. Generic "
            "default time windows and unrelated outcome/timing constraints are "
            "not authoritative trajectory-anchor provenance. Then write "
            "trailing_na_policy={zero_imputation:false, "
            "eligibility_uses_observed_window_count:true, "
            "profile_summaries_ignore_missing:true}. Do not infer the anchor from "
            "trajectory column names. observation_columns must equal (not merely "
            "overlap or add to) every explicitly selected step input from the "
            "declared observation_family, ordered by parsed time bin.\n"
            "- Write cluster_selection.json with criterion, selection_rule "
            "(minimum/maximum/elbow/multi_criteria), direction "
            "(minimize/maximize/not_applicable), selected_n_clusters, at least two "
            "unique candidates [{n_clusters,criterion_value}], and rationale. "
            "Minimum/maximum must select the corresponding finite optimum; elbow "
            "or multi_criteria requires a substantive rationale. Repeat the full "
            "manifest as step_summary.cluster_selection.\n"
            "- trajectory_membership.csv: one row for EVERY locked-cohort id with "
            "observed_window_count, meets_min_observed_windows, "
            "included_in_clustering, exclusion_reason. cluster_assignments.csv: "
            "exactly one row per included id with cluster.\n"
            "- trajectory_profiles.csv: cluster, source_column, "
            "window_start_hours, window_end_hours, summary_statistic, value, "
            "n_observed. cohort_flow.csv: metric,n for input_cohort, "
            "meets_min_observed_windows, excluded_insufficient_windows, and "
            "included_in_clustering. cluster_sizes.csv: cluster,n.\n"
            "- cluster_stability_assignments.csv: resample_id, the declared id "
            "column, reference_cluster, resampled_cluster for at least two "
            "agent-chosen resamples/subsamples. cluster_stability.csv: "
            "resample_id,n_overlap,adjusted_rand_index,clustering_method,"
            "refit_model_id,seed,sampling_method,sample_n,sample_id_hash. Use a "
            "distinct refit_model_id, seed, and sampled membership set for each "
            "reported refit; repeating the same full sample does not establish "
            "resampling stability. Hash sorted sampled id tokens joined with "
            "newline using SHA-256. These are replayable reported-refit fields, "
            "not independent proof that a fit call executed.\n"
        )
        + outcome_contract
        + (
            "Every declared count/value above is independently replayed from the "
            "locked cohort and assignments; fabricated or incomplete tables block."
        )
    )


def _ast_zero(node: ast.AST) -> bool:
    return bool(
        isinstance(node, ast.Constant)
        and isinstance(node.value, (int, float))
        and not isinstance(node.value, bool)
        and float(node.value) == 0.0
    )


def trajectory_zero_imputation_detected(
    script_text: str, *, trajectory_columns: Iterable[str]
) -> bool:
    """Detect zero imputation only on a trajectory-derived expression.

    A step may legitimately fill an unrelated indicator or demographic value
    with zero.  Track simple selector/frame assignments so literal column lists
    and dynamic ``startswith`` selectors receive the same decision without
    turning any ``fillna(0)`` elsewhere in the script into a trajectory error.
    """

    columns = {str(value) for value in trajectory_columns if str(value)}
    if not columns:
        return False
    try:
        tree = ast.parse(str(script_text or ""))
    except SyntaxError:
        return False

    selectors: set[str] = set()
    frames: set[str] = set()

    def references_selector(node: ast.AST) -> bool:
        return any(
            isinstance(child, ast.Name) and child.id in selectors
            for child in ast.walk(node)
        )

    def expression_is_trajectory(node: ast.AST) -> bool:
        if isinstance(node, ast.Name):
            return node.id in frames
        if isinstance(node, ast.Subscript):
            if references_selector(node.slice):
                return True
            return _selector_mentions_trajectory(node.slice, columns)
        if isinstance(node, ast.Call):
            if _selector_mentions_trajectory(node, columns):
                return True
            if isinstance(node.func, ast.Attribute):
                return expression_is_trajectory(node.func.value)
            return bool(node.args and expression_is_trajectory(node.args[0]))
        if isinstance(node, ast.Attribute):
            return expression_is_trajectory(node.value)
        return False

    assignments = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)]
    for _ in range(3):
        changed = False
        for assignment in assignments:
            targets = [
                target.id
                for target in assignment.targets
                if isinstance(target, ast.Name)
            ]
            if not targets:
                continue
            if _selector_mentions_trajectory(assignment.value, columns):
                for target in targets:
                    if target not in selectors:
                        selectors.add(target)
                        changed = True
            if expression_is_trajectory(assignment.value):
                for target in targets:
                    if target not in frames:
                        frames.add(target)
                        changed = True
        if not changed:
            break

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Attribute) and node.func.attr == "fillna":
            fill_value = (
                node.args[0]
                if node.args
                else next(
                    (
                        keyword.value
                        for keyword in node.keywords
                        if keyword.arg == "value"
                    ),
                    None,
                )
            )
            if (
                fill_value is not None
                and _ast_zero(fill_value)
                and expression_is_trajectory(node.func.value)
            ):
                return True
        function_name = (
            node.func.id
            if isinstance(node.func, ast.Name)
            else node.func.attr if isinstance(node.func, ast.Attribute) else ""
        )
        if function_name == "nan_to_num" and node.args:
            nan_keyword = next(
                (keyword.value for keyword in node.keywords if keyword.arg == "nan"),
                None,
            )
            # numpy.nan_to_num defaults nan to 0.0. Explicit non-zero
            # replacement remains outside this narrowly scoped prohibition.
            replaces_nan_with_zero = nan_keyword is None or _ast_zero(nan_keyword)
            if replaces_nan_with_zero and expression_is_trajectory(node.args[0]):
                return True
    return False


def trajectory_script_findings(
    *,
    context: ResearchContext,
    step: Optional[AnalysisStep],
    script_text: str,
) -> list[ValidationFinding]:
    selected = selected_trajectory_variables(
        context=context,
        script_text=script_text,
        step=step,
    )
    if not selected:
        return []
    columns = [variable.name for variable in selected]
    findings: list[ValidationFinding] = []
    if trajectory_zero_imputation_detected(
        script_text,
        trajectory_columns=columns,
    ):
        findings.append(
            ValidationFinding(
                validator="trajectory_representation_contract",
                severity="error",
                message=(
                    "Fixed-window trajectory values were zero-imputed. A trailing "
                    "or unobserved window is not an observed zero state; declare a "
                    "non-zero missingness representation and preserve the observed-"
                    "window membership rule."
                ),
                detail={
                    "step_id": step.step_id if step is not None else None,
                    "trajectory_families": sorted(
                        {
                            variable.fixed_window_trajectory.family
                            for variable in selected
                            if variable.fixed_window_trajectory is not None
                        }
                    ),
                    "selected_trajectory_columns": sorted(
                        variable.name for variable in selected
                    ),
                },
            )
        )
    if trajectory_future_imputation_detected(
        script_text,
        trajectory_columns=columns,
    ):
        findings.append(
            ValidationFinding(
                validator="trajectory_representation_contract",
                severity="error",
                message=(
                    "A fixed-window trajectory uses backward/two-sided filling. "
                    "An earlier window may not consume a future observation; use "
                    "an explicitly ordered, within-entity past-only strategy or "
                    "preserve the missing window."
                ),
                detail={
                    "kind": "trajectory_future_imputation",
                    "step_id": step.step_id if step is not None else None,
                    "selected_trajectory_columns": sorted(columns),
                },
            )
        )
    return findings


def trajectory_future_imputation_detected(
    script_text: str,
    *,
    trajectory_columns: Iterable[str],
) -> bool:
    """Return whether fixed-window values can be filled from future windows."""

    columns = {str(value) for value in trajectory_columns if str(value)}
    if not columns:
        return False
    try:
        tree = ast.parse(str(script_text or ""))
    except SyntaxError:
        return False

    selectors: set[str] = set()
    frames: set[str] = set()

    def references_selector(node: ast.AST) -> bool:
        return any(
            isinstance(child, ast.Name) and child.id in selectors
            for child in ast.walk(node)
        )

    def expression_is_trajectory(node: ast.AST) -> bool:
        if isinstance(node, ast.Name):
            return node.id in frames
        if isinstance(node, ast.Subscript):
            return references_selector(node.slice) or _selector_mentions_trajectory(
                node.slice, columns
            )
        if isinstance(node, ast.Call):
            if _selector_mentions_trajectory(node, columns):
                return True
            if isinstance(node.func, ast.Attribute):
                return expression_is_trajectory(node.func.value)
            return bool(node.args and expression_is_trajectory(node.args[0]))
        if isinstance(node, ast.Attribute):
            return expression_is_trajectory(node.value)
        return False

    assignments = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)]
    for _ in range(3):
        changed = False
        for assignment in assignments:
            targets = [
                target.id
                for target in assignment.targets
                if isinstance(target, ast.Name)
            ]
            if not targets:
                continue
            if _selector_mentions_trajectory(assignment.value, columns):
                for target in targets:
                    if target not in selectors:
                        selectors.add(target)
                        changed = True
            if expression_is_trajectory(assignment.value):
                for target in targets:
                    if target not in frames:
                        frames.add(target)
                        changed = True
        if not changed:
            break

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        method = node.func.attr.lower()
        if not expression_is_trajectory(node.func.value):
            continue
        if method in {"bfill", "backfill", "interpolate"}:
            return True
        if method != "fillna":
            continue
        fill_method = next(
            (
                keyword.value.value.lower()
                for keyword in node.keywords
                if keyword.arg == "method"
                and isinstance(keyword.value, ast.Constant)
                and isinstance(keyword.value.value, str)
            ),
            "",
        )
        if fill_method in {"bfill", "backfill"}:
            return True
    return False


def _contract_error(kind: str, message: str, **detail: Any) -> ValidationFinding:
    return ValidationFinding(
        validator="trajectory_phenotyping_contract",
        severity="error",
        message=message,
        detail={"kind": kind, **detail},
    )


def _token(value: Any) -> Optional[str]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, bool):
        return str(value).lower()
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value).strip()
    if math.isfinite(number) and number.is_integer():
        return str(int(number))
    return str(value).strip()


def _as_int(value: Any, *, minimum: int = 0) -> Optional[int]:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or not number.is_integer() or number < minimum:
        return None
    return int(number)


def _as_float(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _as_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def _read_json_object(path: Path) -> Optional[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _read_csv(path: Path) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _missing_columns(frame: pd.DataFrame, required: set[str]) -> list[str]:
    return sorted(required - {str(column) for column in frame.columns})


def _normalized_method(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")


def _task_contract_anchor_state(
    context: ResearchContext,
) -> tuple[str, Optional[str], list[str]]:
    """Classify trajectory-scoped task anchors as absent, unique, or conflict.

    Only ``relative_to_anchor`` is trajectory alignment evidence. Other parsed
    constraints can describe outcomes, exposure windows, or ordering relative
    to interventions and must never bind the trajectory coordinate system.
    """

    anchors: set[str] = set()
    for constraint in context.temporal_constraints or []:
        if constraint.relation != "relative_to_anchor":
            continue
        anchor = str(constraint.anchor_event or "").strip()
        if anchor:
            anchors.add(anchor)
    ordered = sorted(anchors)
    if not ordered:
        return "absent", None, []
    if len(ordered) == 1:
        return "unique", ordered[0], ordered
    return "conflict", None, ordered


def trajectory_phenotyping_artifact_findings(
    *,
    context: ResearchContext,
    cohort_path: Path,
    step: AnalysisStep,
    out_dir: Path,
    step_summary: Mapping[str, Any],
) -> list[ValidationFinding]:
    """Replay standardized trajectory-phenotyping artifacts from source data.

    The agent owns every scientific choice.  The validator checks only that the
    declared choices agree across artifacts and that memberships, summaries,
    stability metrics, and descriptive outcomes reproduce from row-level data.
    """

    if not trajectory_phenotyping_contract_applies(context=context, step=step):
        return []
    findings: list[ValidationFinding] = []
    outcome_required = trajectory_outcome_contract_declared(step)
    paths = {
        name: out_dir / filename for name, filename in _TRAJECTORY_ARTIFACTS.items()
    }
    if outcome_required:
        paths["outcome"] = out_dir / _OUTCOME_ARTIFACT
    missing_files = sorted(path.name for path in paths.values() if not path.is_file())
    if missing_files:
        return [
            _contract_error(
                "missing_trajectory_artifacts",
                "Trajectory phenotyping is missing replayable standardized artifacts: "
                + ", ".join(missing_files)
                + ".",
                step_id=step.step_id,
                missing_files=missing_files,
            )
        ]

    policy = _read_json_object(paths["policy"])
    if policy is None:
        return [
            _contract_error(
                "invalid_trajectory_policy",
                "trajectory_missingness_policy.json must be a readable JSON object.",
                step_id=step.step_id,
            )
        ]
    try:
        cohort = pd.read_parquet(cohort_path)
    except Exception as exc:
        return [
            _contract_error(
                "trajectory_cohort_unreadable",
                f"Could not replay trajectory phenotyping from the locked cohort: {exc}",
                step_id=step.step_id,
            )
        ]

    id_column = str(policy.get("id_column") or "").strip()
    family = str(policy.get("observation_family") or "").strip()
    observation_columns = policy.get("observation_columns")
    profile_columns = policy.get("profile_columns")
    min_windows = _as_int(policy.get("min_observed_windows"), minimum=1)
    profile_statistic = str(policy.get("profile_summary_statistic") or "").lower()
    method = str(policy.get("clustering_method") or "").strip()
    n_clusters = _as_int(policy.get("n_clusters"), minimum=1)
    time_axis = str(policy.get("time_axis") or "").strip()
    anchor = str(policy.get("anchor") or "").strip()
    anchor_provenance = str(policy.get("anchor_provenance") or "").strip()
    anchor_source = str(policy.get("anchor_source") or "").strip()
    trailing_policy = policy.get("trailing_na_policy")

    policy_problems: list[str] = []
    if not id_column or id_column not in cohort.columns:
        policy_problems.append("id_column is absent from the locked cohort")
    if not family:
        policy_problems.append("observation_family is required")
    if not isinstance(observation_columns, list) or not observation_columns:
        policy_problems.append("observation_columns must be a non-empty list")
    if not isinstance(profile_columns, list) or not profile_columns:
        policy_problems.append("profile_columns must be a non-empty list")
    if min_windows is None:
        policy_problems.append("min_observed_windows must be a positive integer")
    if profile_statistic not in {"mean", "median"}:
        policy_problems.append("profile_summary_statistic must be mean or median")
    if not method:
        policy_problems.append("clustering_method is required")
    if n_clusters is None:
        policy_problems.append("n_clusters must be a positive integer")
    if time_axis != "relative_hours":
        policy_problems.append("time_axis must match the parsed relative_hours axis")
    if not anchor:
        policy_problems.append("anchor is required")
    if anchor_provenance not in {"task_contract", "agent_declared"}:
        policy_problems.append(
            "anchor_provenance must be task_contract or agent_declared"
        )
    if not anchor_source:
        policy_problems.append("anchor_source is required")
    required_trailing = {
        "zero_imputation": False,
        "eligibility_uses_observed_window_count": True,
        "profile_summaries_ignore_missing": True,
    }
    if not isinstance(trailing_policy, dict) or any(
        trailing_policy.get(key) is not expected
        for key, expected in required_trailing.items()
    ):
        policy_problems.append(
            "trailing_na_policy must explicitly preserve missing windows, use "
            "observed-window eligibility, and ignore missing values in profiles"
        )
    if policy_problems:
        return [
            _contract_error(
                "invalid_trajectory_policy",
                "The agent-declared trajectory policy is incomplete or invalid: "
                + "; ".join(policy_problems)
                + ".",
                step_id=step.step_id,
                problems=policy_problems,
            )
        ]

    anchor_state, task_anchor, task_anchors = _task_contract_anchor_state(context)
    task_source = "temporal_constraints.relative_to_anchor"
    if anchor_state == "conflict":
        return [
            _contract_error(
                "trajectory_anchor_contract_conflict",
                "The task declares conflicting relative trajectory anchors; "
                "execution must fail closed until the agent/user resolves one anchor.",
                step_id=step.step_id,
                conflicting_anchors=task_anchors,
            )
        ]
    if anchor_state == "unique" and (
        anchor != task_anchor
        or anchor_provenance != "task_contract"
        or anchor_source != task_source
    ):
        return [
            _contract_error(
                "trajectory_anchor_mismatch",
                "The trajectory anchor/provenance disagrees with the explicit "
                "relative-to-anchor task contract.",
                step_id=step.step_id,
                expected_anchor=task_anchor,
                expected_anchor_provenance="task_contract",
                expected_anchor_source=task_source,
                reported_anchor=anchor,
                reported_anchor_provenance=anchor_provenance,
                reported_anchor_source=anchor_source,
            )
        ]
    if anchor_state == "absent" and anchor_provenance != "agent_declared":
        return [
            _contract_error(
                "trajectory_anchor_provenance_invalid",
                "Without a unique relative-to-anchor task contract, the trajectory "
                "policy must mark the anchor as agent_declared and record its source.",
                step_id=step.step_id,
                reported_anchor=anchor,
                reported_anchor_provenance=anchor_provenance,
                reported_anchor_source=anchor_source,
            )
        ]

    selection_payload = _read_json_object(paths["selection"])
    try:
        selection = ClusterSelectionManifest.model_validate(selection_payload)
    except ValidationError as exc:
        return [
            _contract_error(
                "invalid_cluster_selection_manifest",
                "cluster_selection.json must satisfy the typed candidate-selection "
                "schema with at least two finite candidate values.",
                step_id=step.step_id,
                validation_errors=exc.errors(include_url=False),
            )
        ]
    assert n_clusters is not None
    selection_issues: list[str] = []
    if selection.selected_n_clusters != n_clusters:
        selection_issues.append("selected_n_clusters differs from policy n_clusters")
    selected_value = next(
        item.criterion_value
        for item in selection.candidates
        if item.n_clusters == selection.selected_n_clusters
    )
    candidate_values = [item.criterion_value for item in selection.candidates]
    if (
        selection.candidate_range_boundary_rule
        == "fail_closed_if_selected_at_upper_boundary"
        and selection.selected_n_clusters
        == max(item.n_clusters for item in selection.candidates)
    ):
        return [
            _contract_error(
                "candidate_range_does_not_contain_interior_optimum",
                "The selected criterion optimum is at the frozen candidate-range "
                "upper boundary, so the solution must fail closed without post-hoc "
                "range expansion.",
                step_id=step.step_id,
                reason_code=selection.candidate_range_boundary_reason_code,
                selected_n_clusters=selection.selected_n_clusters,
            )
        ]
    if selection.selection_rule == "minimum" and not math.isclose(
        selected_value,
        min(candidate_values),
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        selection_issues.append("minimum rule did not select the finite minimum")
    if selection.selection_rule == "maximum" and not math.isclose(
        selected_value,
        max(candidate_values),
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        selection_issues.append("maximum rule did not select the finite maximum")
    summary_selection_payload = step_summary.get("cluster_selection")
    try:
        summary_selection = ClusterSelectionManifest.model_validate(
            summary_selection_payload
        )
    except ValidationError:
        summary_selection = None
        selection_issues.append("step_summary.cluster_selection is missing or invalid")
    if summary_selection is not None and (
        summary_selection.model_dump(mode="json") != selection.model_dump(mode="json")
    ):
        selection_issues.append(
            "step_summary.cluster_selection differs from cluster_selection.json"
        )
    if selection_issues:
        return [
            _contract_error(
                "cluster_selection_replay_mismatch",
                "The agent-declared cluster selection does not replay from its "
                "candidate manifest and selected cluster count.",
                step_id=step.step_id,
                issues=selection_issues,
                selection=selection.model_dump(mode="json"),
            )
        ]

    observation_columns = [str(value) for value in observation_columns]
    profile_columns = [str(value) for value in profile_columns]
    variables = {variable.name: variable for variable in context.variables}
    input_names = {str(value) for value in (step.inputs or [])}
    selected_family_columns = sorted(
        (
            variable.name
            for variable in context.variables
            if variable.name in input_names
            and variable.fixed_window_trajectory is not None
            and variable.fixed_window_trajectory.family == family
        ),
        key=lambda column: (
            variables[column].fixed_window_trajectory.window_start_hours,  # type: ignore[union-attr]
            variables[column].fixed_window_trajectory.window_end_hours,  # type: ignore[union-attr]
            column,
        ),
    )
    observation_binding_mismatch = observation_columns != selected_family_columns
    invalid_observation_columns = []
    for column in observation_columns:
        variable = variables.get(column)
        metadata = variable.fixed_window_trajectory if variable is not None else None
        if (
            column not in cohort.columns
            or metadata is None
            or metadata.family != family
        ):
            invalid_observation_columns.append(column)
    invalid_profile_columns = [
        column
        for column in profile_columns
        if column not in cohort.columns
        or column not in variables
        or variables[column].fixed_window_trajectory is None
    ]
    observation_bins = [
        (
            variables[column].fixed_window_trajectory.window_start_hours,
            variables[column].fixed_window_trajectory.window_end_hours,
        )
        for column in observation_columns
        if column not in invalid_observation_columns
    ]
    if (
        invalid_observation_columns
        or observation_binding_mismatch
        or invalid_profile_columns
        or len(set(observation_columns)) != len(observation_columns)
        or len(set(profile_columns)) != len(profile_columns)
        or len(set(observation_bins)) != len(observation_bins)
        or min_windows > len(observation_columns)
    ):
        return [
            _contract_error(
                "invalid_trajectory_columns",
                "The declared trajectory columns/time bins cannot be replayed from "
                "the context and locked cohort.",
                step_id=step.step_id,
                invalid_observation_columns=invalid_observation_columns,
                observation_binding_policy="ordered_equality",
                step_selected_family_columns=selected_family_columns,
                declared_observation_columns=observation_columns,
                invalid_profile_columns=invalid_profile_columns,
                min_observed_windows=min_windows,
                observation_column_count=len(observation_columns),
            )
        ]

    analysis_script = out_dir.parent / "analysis.py"
    if analysis_script.is_file():
        try:
            script_text = analysis_script.read_text(encoding="utf-8")
        except OSError:
            script_text = ""
        if trajectory_zero_imputation_detected(
            script_text,
            trajectory_columns=observation_columns,
        ):
            findings.append(
                _contract_error(
                    "trajectory_zero_imputation",
                    "The executed trajectory script contains zero imputation; "
                    "the declared no-zero policy is false.",
                    step_id=step.step_id,
                )
            )

    membership = _read_csv(paths["membership"])
    assignments = _read_csv(paths["assignments"])
    profiles = _read_csv(paths["profiles"])
    flow = _read_csv(paths["flow"])
    sizes = _read_csv(paths["sizes"])
    stability = _read_csv(paths["stability"])
    stability_assignments = _read_csv(paths["stability_assignments"])
    outcome = _read_csv(paths["outcome"]) if outcome_required else None
    artifact_frames = {
        "trajectory_membership.csv": membership,
        "cluster_assignments.csv": assignments,
        "trajectory_profiles.csv": profiles,
        "cohort_flow.csv": flow,
        "cluster_sizes.csv": sizes,
        "cluster_stability.csv": stability,
        "cluster_stability_assignments.csv": stability_assignments,
    }
    if outcome_required:
        artifact_frames[_OUTCOME_ARTIFACT] = outcome
    unreadable = sorted(
        name for name, frame in artifact_frames.items() if frame is None
    )
    if unreadable:
        return findings + [
            _contract_error(
                "unreadable_trajectory_artifacts",
                "Could not read standardized trajectory artifacts: "
                + ", ".join(unreadable)
                + ".",
                step_id=step.step_id,
            )
        ]

    assert membership is not None
    assert assignments is not None
    assert profiles is not None
    assert flow is not None
    assert sizes is not None
    assert stability is not None
    assert stability_assignments is not None
    if outcome_required:
        assert outcome is not None

    membership_required = {
        id_column,
        "observed_window_count",
        "meets_min_observed_windows",
        "included_in_clustering",
        "exclusion_reason",
    }
    assignments_required = {id_column, "cluster"}
    missing_schema = {
        "trajectory_membership.csv": _missing_columns(membership, membership_required),
        "cluster_assignments.csv": _missing_columns(assignments, assignments_required),
        "trajectory_profiles.csv": _missing_columns(
            profiles,
            {
                "cluster",
                "source_column",
                "window_start_hours",
                "window_end_hours",
                "summary_statistic",
                "value",
                "n_observed",
            },
        ),
        "cohort_flow.csv": _missing_columns(flow, {"metric", "n"}),
        "cluster_sizes.csv": _missing_columns(sizes, {"cluster", "n"}),
        "cluster_stability.csv": _missing_columns(
            stability,
            {
                "resample_id",
                "n_overlap",
                "adjusted_rand_index",
                "clustering_method",
                "refit_model_id",
                "seed",
                "sampling_method",
                "sample_n",
                "sample_id_hash",
            },
        ),
        "cluster_stability_assignments.csv": _missing_columns(
            stability_assignments,
            {"resample_id", id_column, "reference_cluster", "resampled_cluster"},
        ),
    }
    missing_schema = {name: cols for name, cols in missing_schema.items() if cols}
    if missing_schema:
        return findings + [
            _contract_error(
                "trajectory_artifact_schema_missing",
                "Standardized trajectory artifacts are missing required columns.",
                step_id=step.step_id,
                missing_columns=missing_schema,
            )
        ]

    cohort_ids = cohort[id_column].map(_token)
    membership_ids = membership[id_column].map(_token)
    if (
        cohort_ids.isna().any()
        or membership_ids.isna().any()
        or cohort_ids.duplicated().any()
        or membership_ids.duplicated().any()
        or set(cohort_ids) != set(membership_ids)
    ):
        return findings + [
            _contract_error(
                "trajectory_membership_id_mismatch",
                "trajectory_membership.csv must contain every locked-cohort id "
                "exactly once and no foreign ids.",
                step_id=step.step_id,
                cohort_id_count=int(cohort_ids.nunique()),
                membership_id_count=int(membership_ids.nunique()),
            )
        ]

    expected_observed = cohort[observation_columns].notna().sum(axis=1).astype(int)
    expected_by_id = dict(zip(cohort_ids, expected_observed, strict=True))
    observed_reported = pd.to_numeric(
        membership["observed_window_count"], errors="coerce"
    )
    meets_reported = membership["meets_min_observed_windows"].map(_as_bool)
    included_reported = membership["included_in_clustering"].map(_as_bool)
    membership_issues: list[dict[str, Any]] = []
    included_ids: set[str] = set()
    for index, member_id in membership_ids.items():
        expected_count = expected_by_id[member_id]
        reported_count = _as_int(observed_reported.loc[index], minimum=0)
        expected_meets = bool(expected_count >= min_windows)
        reported_meets = meets_reported.loc[index]
        reported_included = included_reported.loc[index]
        reason = _token(membership.loc[index, "exclusion_reason"]) or ""
        if reported_count != expected_count:
            membership_issues.append(
                {"id": member_id, "issue": "observed_window_count_mismatch"}
            )
        if reported_meets != expected_meets:
            membership_issues.append(
                {"id": member_id, "issue": "threshold_flag_mismatch"}
            )
        if reported_included != expected_meets:
            membership_issues.append(
                {"id": member_id, "issue": "included_flag_mismatch"}
            )
        if expected_meets:
            included_ids.add(member_id)
        elif not reason:
            membership_issues.append(
                {"id": member_id, "issue": "missing_exclusion_reason"}
            )
        if len(membership_issues) >= 20:
            break
    if membership_issues:
        return findings + [
            _contract_error(
                "trajectory_membership_replay_mismatch",
                "The agent-declared observed-window cohort membership does not "
                "replay from the locked cohort and chosen threshold.",
                step_id=step.step_id,
                issues=membership_issues,
            )
        ]

    assignment_ids = assignments[id_column].map(_token)
    assignment_clusters = assignments["cluster"].map(_token)
    if (
        assignment_ids.isna().any()
        or assignment_clusters.isna().any()
        or assignment_ids.duplicated().any()
        or set(assignment_ids) != included_ids
    ):
        return findings + [
            _contract_error(
                "cluster_assignments_membership_mismatch",
                "cluster_assignments.csv must contain exactly one cluster label "
                "for every replayed included id and no other ids.",
                step_id=step.step_id,
                expected_assignment_n=len(included_ids),
                reported_assignment_n=int(assignment_ids.nunique()),
                foreign_ids=sorted(set(assignment_ids.dropna()) - included_ids)[:20],
                missing_ids=sorted(included_ids - set(assignment_ids.dropna()))[:20],
            )
        ]
    cluster_by_id = dict(zip(assignment_ids, assignment_clusters, strict=True))
    cluster_counts = assignment_clusters.value_counts().to_dict()
    if len(cluster_counts) != n_clusters:
        findings.append(
            _contract_error(
                "cluster_count_mismatch",
                "The declared n_clusters does not equal the number of row-level "
                "assignment labels.",
                step_id=step.step_id,
                declared_n_clusters=n_clusters,
                observed_n_clusters=len(cluster_counts),
            )
        )

    size_labels = sizes["cluster"].map(_token)
    duplicate_size_labels = sorted(
        set(size_labels[size_labels.duplicated(keep=False)].dropna())
    )
    size_rows = dict(
        zip(
            size_labels,
            (_as_int(value, minimum=0) for value in sizes["n"]),
            strict=True,
        )
    )
    if (
        size_labels.isna().any()
        or duplicate_size_labels
        or set(size_labels) != set(cluster_counts)
        or size_rows != cluster_counts
    ):
        findings.append(
            _contract_error(
                "cluster_sizes_mismatch",
                "cluster_sizes.csv must contain exactly one row per assigned "
                "cluster and reproduce row-level assignment counts.",
                step_id=step.step_id,
                expected=cluster_counts,
                reported=size_rows,
                duplicate_clusters=duplicate_size_labels,
                extra_clusters=sorted(set(size_labels.dropna()) - set(cluster_counts)),
            )
        )

    expected_flow = {
        "input_cohort": len(cohort),
        "meets_min_observed_windows": len(included_ids),
        "excluded_insufficient_windows": len(cohort) - len(included_ids),
        "included_in_clustering": len(included_ids),
    }
    flow_metrics = flow["metric"].map(_token)
    duplicate_flow_metrics = sorted(
        set(flow_metrics[flow_metrics.duplicated(keep=False)].dropna())
    )
    flow_rows = dict(
        zip(
            flow_metrics,
            (_as_int(value, minimum=0) for value in flow["n"]),
            strict=True,
        )
    )
    if (
        flow_metrics.isna().any()
        or duplicate_flow_metrics
        or set(flow_metrics) != set(expected_flow)
        or flow_rows != expected_flow
    ):
        findings.append(
            _contract_error(
                "trajectory_cohort_flow_mismatch",
                "cohort_flow.csv must contain each standard flow row exactly "
                "once and reproduce from the locked cohort, chosen minimum-window "
                "threshold, and assignments.",
                step_id=step.step_id,
                expected=expected_flow,
                reported=flow_rows,
                duplicate_metrics=duplicate_flow_metrics,
                extra_metrics=sorted(set(flow_metrics.dropna()) - set(expected_flow)),
            )
        )

    profile_rows: dict[tuple[str, str], Mapping[str, Any]] = {}
    duplicate_profiles: list[tuple[str, str]] = []
    for _, row in profiles.iterrows():
        key = (_token(row["cluster"]), str(row["source_column"] or "").strip())
        if None in key or key in profile_rows:
            duplicate_profiles.append(key)  # type: ignore[arg-type]
        else:
            profile_rows[key] = row
    cohort_indexed = cohort.assign(__id_token=cohort_ids).set_index("__id_token")
    expected_profile_keys = {
        (cluster, column) for cluster in cluster_counts for column in profile_columns
    }
    profile_issues: list[dict[str, Any]] = []
    if set(profile_rows) != expected_profile_keys or duplicate_profiles:
        profile_issues.append(
            {
                "issue": "profile_row_keys_mismatch",
                "missing": sorted(expected_profile_keys - set(profile_rows))[:20],
                "extra": sorted(set(profile_rows) - expected_profile_keys)[:20],
                "duplicates": duplicate_profiles[:20],
            }
        )
    for cluster, column in sorted(expected_profile_keys & set(profile_rows)):
        row = profile_rows[(cluster, column)]
        metadata = variables[column].fixed_window_trajectory
        assert metadata is not None
        cluster_ids = [
            member_id for member_id, label in cluster_by_id.items() if label == cluster
        ]
        values = pd.to_numeric(
            cohort_indexed.loc[cluster_ids, column], errors="coerce"
        ).dropna()
        expected_value = (
            float(values.mean())
            if profile_statistic == "mean"
            else float(values.median())
        )
        reported_value = _as_float(row["value"])
        reported_n = _as_int(row["n_observed"], minimum=0)
        if (
            str(row["summary_statistic"] or "").strip().lower() != profile_statistic
            or _as_float(row["window_start_hours"]) != metadata.window_start_hours
            or _as_float(row["window_end_hours"]) != metadata.window_end_hours
            or reported_n != len(values)
            or reported_value is None
            or not math.isclose(
                reported_value, expected_value, rel_tol=1e-6, abs_tol=1e-8
            )
        ):
            profile_issues.append(
                {
                    "issue": "profile_value_mismatch",
                    "cluster": cluster,
                    "source_column": column,
                    "expected_value": expected_value,
                    "reported_value": reported_value,
                    "expected_n": len(values),
                    "reported_n": reported_n,
                }
            )
        if len(profile_issues) >= 20:
            break
    if profile_issues:
        findings.append(
            _contract_error(
                "trajectory_profiles_mismatch",
                "trajectory_profiles.csv is incomplete, unordered, or disagrees "
                "with source-window summaries within assigned clusters.",
                step_id=step.step_id,
                issues=profile_issues,
            )
        )

    findings.extend(
        _stability_findings(
            step=step,
            id_column=id_column,
            method=method,
            n_clusters=n_clusters,
            cluster_by_id=cluster_by_id,
            stability=stability,
            stability_assignments=stability_assignments,
        )
    )
    if outcome_required:
        assert outcome is not None
        findings.extend(
            _outcome_findings(
                context=context,
                step=step,
                policy=policy,
                cohort_indexed=cohort_indexed,
                cluster_by_id=cluster_by_id,
                cluster_counts=cluster_counts,
                outcome=outcome,
            )
        )

    summary_k = _as_int(
        step_summary.get("n_clusters", step_summary.get("cluster_count")), minimum=1
    )
    summary_method = str(
        step_summary.get("clustering_method") or step_summary.get("algorithm") or ""
    ).strip()
    summary_min_windows = _as_int(step_summary.get("min_observed_windows"), minimum=1)
    if (
        summary_k != n_clusters
        or _normalized_method(summary_method) != _normalized_method(method)
        or summary_min_windows != min_windows
    ):
        findings.append(
            _contract_error(
                "trajectory_summary_declaration_mismatch",
                "step_summary.json must repeat the agent-declared clustering method, "
                "k, and minimum observed-window threshold exactly.",
                step_id=step.step_id,
                policy_method=method,
                summary_method=summary_method,
                policy_n_clusters=n_clusters,
                summary_n_clusters=summary_k,
                policy_min_observed_windows=min_windows,
                summary_min_observed_windows=summary_min_windows,
            )
        )
    return findings


def _sample_id_hash(ids: Iterable[str]) -> str:
    payload = "\n".join(sorted(str(value) for value in ids)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _stability_findings(
    *,
    step: AnalysisStep,
    id_column: str,
    method: str,
    n_clusters: int,
    cluster_by_id: Mapping[str, str],
    stability: pd.DataFrame,
    stability_assignments: pd.DataFrame,
) -> list[ValidationFinding]:
    from sklearn.metrics import adjusted_rand_score

    findings: list[ValidationFinding] = []
    summary_ids = stability["resample_id"].map(_token)
    assignment_resample_ids = stability_assignments["resample_id"].map(_token)
    refit_ids = stability["refit_model_id"].map(_token)
    seeds = stability["seed"].map(lambda value: _as_int(value, minimum=0))
    sampling_methods = stability["sampling_method"].map(_token)
    declared_sample_hashes = stability["sample_id_hash"].map(_token)
    resample_ids = sorted(set(summary_ids.dropna()))
    if (
        len(resample_ids) < 2
        or summary_ids.isna().any()
        or summary_ids.duplicated().any()
        or set(assignment_resample_ids.dropna()) != set(resample_ids)
    ):
        return [
            _contract_error(
                "cluster_stability_resamples_invalid",
                "Stability evidence must contain at least two uniquely summarized "
                "resamples and matching row-level assignment evidence.",
                step_id=step.step_id,
                summary_resample_ids=resample_ids,
                assignment_resample_ids=sorted(set(assignment_resample_ids.dropna())),
            )
        ]
    if (
        refit_ids.isna().any()
        or refit_ids.duplicated().any()
        or seeds.isna().any()
        or seeds.duplicated().any()
        or sampling_methods.isna().any()
        or declared_sample_hashes.isna().any()
        or declared_sample_hashes.duplicated().any()
    ):
        return [
            _contract_error(
                "cluster_stability_provenance_invalid",
                "Replayable reported refit evidence requires a distinct "
                "refit_model_id, seed, and sampled membership hash per resample, "
                "plus a non-empty sampling_method. This validates reported "
                "assignment/ARI consistency; it does not independently prove a "
                "model fit call occurred.",
                step_id=step.step_id,
                refit_model_ids=list(refit_ids),
                seeds=list(seeds),
                sampling_methods=list(sampling_methods),
                sample_id_hashes=list(declared_sample_hashes),
            )
        ]

    summary_by_id = {_token(row["resample_id"]): row for _, row in stability.iterrows()}
    issues: list[dict[str, Any]] = []
    for resample_id in resample_ids:
        mask = assignment_resample_ids == resample_id
        rows = stability_assignments.loc[mask].copy()
        row_ids = rows[id_column].map(_token)
        reference = rows["reference_cluster"].map(_token)
        resampled = rows["resampled_cluster"].map(_token)
        if (
            row_ids.isna().any()
            or reference.isna().any()
            or resampled.isna().any()
            or row_ids.duplicated().any()
            or len(rows) < 2
            or not set(row_ids).issubset(cluster_by_id)
        ):
            issues.append(
                {"resample_id": resample_id, "issue": "invalid_assignment_rows"}
            )
            continue
        expected_reference = row_ids.map(cluster_by_id)
        if not reference.reset_index(drop=True).equals(
            expected_reference.reset_index(drop=True)
        ):
            issues.append(
                {"resample_id": resample_id, "issue": "reference_labels_forged"}
            )
            continue
        reference_label_count = len(set(reference))
        resampled_label_count = len(set(resampled))
        if not (
            2 <= reference_label_count <= n_clusters
            and 2 <= resampled_label_count <= n_clusters
        ):
            issues.append(
                {
                    "resample_id": resample_id,
                    "issue": "stability_labels_invalid",
                    "declared_n_clusters": n_clusters,
                    "reference_label_count": reference_label_count,
                    "resampled_label_count": resampled_label_count,
                }
            )
            continue
        recomputed = float(adjusted_rand_score(reference, resampled))
        summary = summary_by_id[resample_id]
        reported_ari = _as_float(summary["adjusted_rand_index"])
        reported_n = _as_int(summary["n_overlap"], minimum=0)
        reported_sample_n = _as_int(summary["sample_n"], minimum=0)
        reported_sample_hash = str(summary["sample_id_hash"] or "").strip().lower()
        expected_sample_hash = _sample_id_hash(row_ids)
        reported_method = str(summary["clustering_method"] or "").strip()
        if (
            reported_n != len(rows)
            or reported_sample_n != len(rows)
            or reported_sample_hash != expected_sample_hash
            or reported_ari is None
            or not math.isclose(reported_ari, recomputed, rel_tol=1e-8, abs_tol=1e-8)
            or _normalized_method(reported_method) != _normalized_method(method)
        ):
            issues.append(
                {
                    "resample_id": resample_id,
                    "issue": "stability_summary_mismatch",
                    "expected_n_overlap": len(rows),
                    "reported_n_overlap": reported_n,
                    "expected_sample_n": len(rows),
                    "reported_sample_n": reported_sample_n,
                    "expected_sample_id_hash": expected_sample_hash,
                    "reported_sample_id_hash": reported_sample_hash,
                    "expected_adjusted_rand_index": recomputed,
                    "reported_adjusted_rand_index": reported_ari,
                    "expected_method": method,
                    "reported_method": reported_method,
                }
            )
        if len(issues) >= 20:
            break
    if issues:
        findings.append(
            _contract_error(
                "cluster_stability_replay_mismatch",
                "Cluster-stability metrics do not replay from the agent-produced "
                "reference/resampled labels.",
                step_id=step.step_id,
                issues=issues,
            )
        )
    return findings


def _outcome_findings(
    *,
    context: ResearchContext,
    step: AnalysisStep,
    policy: Mapping[str, Any],
    cohort_indexed: pd.DataFrame,
    cluster_by_id: Mapping[str, str],
    cluster_counts: Mapping[str, int],
    outcome: pd.DataFrame,
) -> list[ValidationFinding]:
    target = str(context.target_outcome or "").strip()
    if not target or target not in cohort_indexed.columns:
        return [
            _contract_error(
                "trajectory_outcome_not_replayable",
                "A descriptive outcome-by-cluster artifact was declared, but the "
                "target outcome is absent from the locked cohort/context.",
                step_id=step.step_id,
                target_outcome=target or None,
            )
        ]
    required_base = {"cluster", "n", "outcome_n"}
    if _missing_columns(outcome, required_base):
        return [
            _contract_error(
                "outcome_by_cluster_schema_missing",
                "outcome_by_cluster.csv lacks cluster, n, or outcome_n.",
                step_id=step.step_id,
                missing_columns=_missing_columns(outcome, required_base),
            )
        ]

    target_values = pd.to_numeric(cohort_indexed[target], errors="coerce")
    observed_levels = set(target_values.dropna().unique().tolist())
    binary = observed_levels.issubset({0, 1}) and bool(observed_levels)
    if binary:
        missing = _missing_columns(outcome, {"event_n", "outcome_rate"})
        summary_statistic = None
    else:
        missing = _missing_columns(outcome, {"summary_statistic", "value"})
        summary_statistic = str(policy.get("outcome_summary_statistic") or "").lower()
        if summary_statistic not in {"mean", "median"}:
            missing.append("policy.outcome_summary_statistic(mean|median)")
    if missing:
        return [
            _contract_error(
                "outcome_by_cluster_schema_missing",
                "outcome_by_cluster.csv lacks replayable descriptive outcome fields.",
                step_id=step.step_id,
                missing_columns=sorted(set(missing)),
            )
        ]

    reported_rows: dict[str, Mapping[str, Any]] = {}
    duplicate_clusters: list[str] = []
    for _, row in outcome.iterrows():
        cluster = _token(row["cluster"])
        if cluster is None or cluster in reported_rows:
            duplicate_clusters.append(str(cluster))
        else:
            reported_rows[cluster] = row
    issues: list[dict[str, Any]] = []
    if set(reported_rows) != set(cluster_counts) or duplicate_clusters:
        issues.append(
            {
                "issue": "outcome_cluster_rows_mismatch",
                "missing": sorted(set(cluster_counts) - set(reported_rows)),
                "extra": sorted(set(reported_rows) - set(cluster_counts)),
                "duplicates": duplicate_clusters,
            }
        )
    for cluster in sorted(set(reported_rows) & set(cluster_counts)):
        row = reported_rows[cluster]
        ids = [
            member_id for member_id, label in cluster_by_id.items() if label == cluster
        ]
        values = target_values.reindex(ids).dropna()
        reported_n = _as_int(row["n"], minimum=0)
        reported_outcome_n = _as_int(row["outcome_n"], minimum=0)
        issue: dict[str, Any] = {
            "cluster": cluster,
            "expected_n": cluster_counts[cluster],
            "reported_n": reported_n,
            "expected_outcome_n": len(values),
            "reported_outcome_n": reported_outcome_n,
        }
        mismatch = reported_n != cluster_counts[cluster] or reported_outcome_n != len(
            values
        )
        if binary:
            expected_events = int(values.sum())
            expected_rate = expected_events / len(values) if len(values) else math.nan
            reported_events = _as_int(row["event_n"], minimum=0)
            reported_rate = _as_float(row["outcome_rate"])
            issue.update(
                {
                    "expected_event_n": expected_events,
                    "reported_event_n": reported_events,
                    "expected_outcome_rate": expected_rate,
                    "reported_outcome_rate": reported_rate,
                }
            )
            mismatch = (
                mismatch
                or reported_events != expected_events
                or (
                    reported_rate is None
                    or not math.isclose(
                        reported_rate, expected_rate, rel_tol=1e-8, abs_tol=1e-8
                    )
                )
            )
        else:
            reported_statistic = str(row["summary_statistic"] or "").lower()
            expected_value = (
                float(values.mean())
                if summary_statistic == "mean"
                else float(values.median())
            )
            reported_value = _as_float(row["value"])
            issue.update(
                {
                    "expected_summary_statistic": summary_statistic,
                    "reported_summary_statistic": reported_statistic,
                    "expected_value": expected_value,
                    "reported_value": reported_value,
                }
            )
            mismatch = (
                mismatch
                or reported_statistic != summary_statistic
                or (
                    reported_value is None
                    or not math.isclose(
                        reported_value, expected_value, rel_tol=1e-8, abs_tol=1e-8
                    )
                )
            )
        if mismatch:
            issue["issue"] = "outcome_value_mismatch"
            issues.append(issue)
    if not issues:
        return []
    return [
        _contract_error(
            "outcome_by_cluster_replay_mismatch",
            "The descriptive outcome-by-cluster artifact disagrees with the "
            "locked cohort and row-level assignments.",
            step_id=step.step_id,
            target_outcome=target,
            issues=issues[:20],
        )
    ]


__all__ = [
    "TRAJECTORY_PHENOTYPING_REQUIRED_OUTPUTS",
    "infer_fixed_window_trajectory_metadata",
    "is_continuous_trajectory_representation",
    "selected_trajectory_variables",
    "trajectory_phenotyping_artifact_findings",
    "trajectory_phenotyping_code_contract",
    "trajectory_phenotyping_contract_applies",
    "trajectory_script_findings",
    "trajectory_future_imputation_detected",
    "trajectory_zero_imputation_detected",
]
