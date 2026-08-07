"""Generic variable-kind × analytical-method compatibility matrix.

This module encodes statistical / clinical truths about *kinds* of
variables (ordinal, binary, count, right-skewed continuous) and the
method families that are inappropriate for each. It is consumed by
:func:`render_variable_constraints` to produce a self-review checklist
for method-selecting Planner/Coder prompts **before** generated code reaches
the deterministic post-hoc validators in ``audits/``. Evidence-interpreting
Analyzer/Writer prompts receive a smaller role-scoped context because they do
not choose or execute an analytical method.

Design constraints
------------------

* **Not case-specific.** Entries are keyed by variable *kind*
  (``ordinal``, ``binary``, ``count``, ``right_skewed_continuous``)
  rather than by concept name. Adding a new kind extends the matrix;
  adding a new benchmark question does not. The compatibility check
  for, say, ``gcs`` follows automatically from
  ``ConceptDescriptor.is_ordinal=True``; no entry for "gcs" exists
  here.
* **Pre-flight, not post-hoc.** Validators under ``audits/`` catch
  these issues after the code runs. The constraints rendered here
  give the LLM the same information *before* it writes code, which
  reduces the rate at which the validator has to fire.
* **Extensible.** A new ``VariableRole`` or ``AggregationRule`` is
  classified by :func:`_variable_kind` and inherits any matching
  matrix entry. Tests under
  ``tests/research_agent/test_method_compatibility.py`` lock the
  expected behaviour for the current enum set.

This module imports only from :mod:`easyicu.research_agent.schema`.
No prompts here. No LLM calls here.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from ..schema import (
    AggregationRule,
    AnalysisStep,
    ConceptDescriptor,
    ResearchContext,
    VariableRole,
)
from ..audits.patterns import (
    _DISTANCE_BASED_ESTIMATORS,
    _LINEAR_PCA_ESTIMATORS,
)
from ..trajectory.contract import (
    is_continuous_trajectory_representation,
    selected_trajectory_variables,
    trajectory_zero_imputation_detected,
)
from ..trajectory.plan_contract import trajectory_step_roles

# ---------------------------------------------------------------------------
# Matrix
# ---------------------------------------------------------------------------

# Each entry: lowercase method-name fragments to forbid + preferred
# replacements + rationale. Patterns are matched against the LLM's
# *written code or method-name strings* in the system prompt; the
# matrix is also re-used by post-hoc validators when downstream
# coverage is added.

#: Method-name substrings that misuse a given variable kind, with the
#: clinically / statistically preferred alternatives and a short
#: rationale. Keys are stable, non-case-specific variable kinds.
#: Distance-based / continuous-PCA estimator names — single source of
#: truth lives in ``audits/patterns.py`` so the pre-flight checklist
#: rendered to the coder and the post-hoc validator that catches what
#: slips through can never drift out of sync. ``_DISTANCE_BASED_ESTIMATORS``
#: enumerates KMeans, MiniBatchKMeans, AgglomerativeClustering, DBSCAN,
#: HDBSCAN, Birch, SpectralClustering, MeanShift, OPTICS, NearestNeighbors
#: and the K-nearest-neighbours classifier/regressor families.
_ORDINAL_FORBIDDEN_PATTERNS = (
    *tuple(s.lower() for s in _DISTANCE_BASED_ESTIMATORS),
    *tuple(s.lower() for s in _LINEAR_PCA_ESTIMATORS),
    "kmeans",
    "k-means",
    "k_means",
    "k_neighbors",
    "euclidean_distance",
    "manhattan_distance",
    "tsne",
    "t-sne",
    "umap.umap",
)


FORBIDDEN_METHOD_BY_KIND: Dict[str, Dict[str, object]] = {
    "ordinal": {
        "forbidden_patterns": _ORDINAL_FORBIDDEN_PATTERNS,
        "preferred": (
            "spearman_correlation",
            "kendall_tau",
            "ordinal_logistic_regression",
            "rank_sum_test",
            "chi_squared_with_grouping",
        ),
        "rationale": (
            "Ordinal variables have rank order but unequal intervals; "
            "Euclidean / continuous-PCA / k-means / k-NN treat ranks as "
            "interval data and misrepresent the metric."
        ),
    },
    "binary": {
        "forbidden_patterns": (
            "linearregression",
            "linear_regression(",
            "ols(",
            "linear regression",
            "kmeans on binary",
        ),
        "preferred": (
            "logistic_regression",
            "binomial_glm",
            "fisher_exact",
        ),
        "rationale": (
            "Binary outcomes need a logistic / binomial link; ordinary least "
            "squares on 0/1 gives invalid coefficients, predictions outside "
            "[0,1] and incorrect confidence intervals."
        ),
    },
    "count": {
        "forbidden_patterns": (
            "normalfit",
            "gaussian_glm",
            "linearregression",
            "linear_regression(",
            "ols(",
        ),
        "preferred": (
            "poisson_regression",
            "negative_binomial_regression",
            "zero_inflated_poisson",
        ),
        "rationale": (
            "Count outcomes are non-negative integers, frequently overdispersed; "
            "Gaussian models can predict negative counts and underestimate "
            "variance."
        ),
    },
    "right_skewed_continuous": {
        "forbidden_patterns": (
            "report mean",
            "report mean and sd",
            ".mean()",
            "np.mean(",
            "numpy.mean(",
            "report mean ± sd",
        ),
        "preferred": (
            "median_iqr",
            "log_transform_then_mean",
            "geometric_mean",
        ),
        "rationale": (
            "Heavily right-skewed lab values (lactate, bilirubin, peak "
            "creatinine, troponin) have means dominated by outliers; report "
            "median (IQR) or log-transform before mean / SD."
        ),
    },
}

# Variables in these LAB names are conventionally right-skewed and
# should map to the ``right_skewed_continuous`` kind irrespective of
# their dtype. This list intentionally lives next to the matrix so a
# reviewer can audit it in one place.
_RIGHT_SKEWED_LAB_NAMES = frozenset(
    {
        "lact",
        "lactate",
        "lactate_max",
        "lactate_peak",
        "bili",
        "bilirubin",
        "tbili",
        "total_bilirubin",
        "crea_peak",
        "creatinine_peak",
        "alt",
        "ast",
        "ggt",
        "troponin",
        "trop",
        "ck",
        "creatine_kinase",
        "ddimer",
        "d_dimer",
        "ferritin",
        "ldh",
    }
)


# Pairwise metrics such as silhouette have quadratic time/memory behaviour.
# Full-data fitting and label assignment remain scientifically desirable; only
# the diagnostic pairwise evaluation is sampled once the execution cohort is
# large.  These case-neutral constants are deliberately owned beside the
# prompt and AST enforcement below so the two surfaces cannot drift.
PAIRWISE_EVALUATION_FULL_COHORT_MAX_ROWS = 10_000
PAIRWISE_EVALUATION_MAX_SAMPLE_SIZE = 5_000


# ---------------------------------------------------------------------------
# Per-variable kind classification
# ---------------------------------------------------------------------------


#: Variable-name suffix / prefix tokens that strongly imply a binary
#: indicator regardless of dtype or allowed_aggregations. Kept short
#: and concept-neutral; extend when a new generic naming convention
#: appears (e.g. ``_any``, ``has_``).
_BINARY_NAME_TOKENS = (
    "death",
    "_indicator",
    "_flag",
    "_any",
    "_any_24h",
    "_present",
    "is_",
    "has_",
)


def _variable_kind(var: ConceptDescriptor) -> Optional[str]:
    """Map a :class:`ConceptDescriptor` to a matrix key.

    Returns ``None`` when no constraint applies (e.g. id / time / meta
    columns and unconstrained continuous variables).

    Ordering note: ordinal first, then binary (including the OUTCOME-role
    naming heuristic), then count, then right-skewed continuous. Binary
    must beat count because an indicator column with SUM in its
    allowed_aggregations (used for cohort totals) is still semantically
    binary and a Gaussian / Poisson model on it is wrong in different
    ways than a logistic model is right.
    """
    if is_continuous_trajectory_representation(var):
        return None
    if var.is_ordinal or var.role == VariableRole.ORDINAL_SCORE:
        return "ordinal"
    dtype = (var.dtype or "").lower()
    if "bool" in dtype:
        return "binary"
    if var.valid_range and len(var.valid_range) == 2:
        lo, hi = var.valid_range
        try:
            lo_i = int(lo)
            hi_i = int(hi)
        except (TypeError, ValueError):
            lo_i, hi_i = None, None
        if lo_i == 0 and hi_i == 1 and "int" in dtype:
            return "binary"
    # OUTCOME / INTERVENTION roles whose names match a generic binary
    # convention. Catches e.g. ``death``, ``vaso_any_24h``, ``mech_vent``
    # without enumerating the concepts themselves.
    if var.role in (VariableRole.OUTCOME, VariableRole.INTERVENTION):
        name_l = var.name.lower()
        if any(tok in name_l for tok in _BINARY_NAME_TOKENS):
            return "binary"
        if name_l in {"vaso", "mech_vent", "rrt", "vent", "pressor"}:
            return "binary"
    if (
        var.role == VariableRole.OUTCOME
        and "int" in dtype
        and AggregationRule.SUM in (var.allowed_aggregations or [])
    ):
        return "count"
    if var.role == VariableRole.LAB and var.name.lower() in _RIGHT_SKEWED_LAB_NAMES:
        return "right_skewed_continuous"
    return None


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def variable_kind_constraints(
    variables: Sequence[ConceptDescriptor],
    *,
    outcome_names: Sequence[str] = (),
) -> List[Dict[str, object]]:
    """Return a list of per-variable constraint dicts (machine-readable)."""
    declared_outcomes = {
        str(name or "").strip().casefold()
        for name in outcome_names
        if str(name).strip()
    }
    out: List[Dict[str, object]] = []
    for var in variables:
        kind = _variable_kind(var)
        if not kind:
            continue
        if kind in {"binary", "count"} and not (
            var.role == VariableRole.OUTCOME
            or var.name.strip().casefold() in declared_outcomes
        ):
            continue
        rule = FORBIDDEN_METHOD_BY_KIND.get(kind)
        if not rule:
            continue
        out.append(
            {
                "variable": var.name,
                "kind": kind,
                "forbidden_patterns": rule["forbidden_patterns"],
                "preferred": rule["preferred"],
                "rationale": rule["rationale"],
            }
        )
    return out


def render_variable_constraints(
    context: ResearchContext,
    *,
    compact: bool = False,
) -> str:
    """Return a human-readable self-review block for agent system prompts.

    Empty string when no variable in ``context.variables`` triggers a
    matrix entry; in that case the agent prompts are unchanged. This
    keeps the addition invisible for cohorts that don't carry any
    ordinal / count / binary / right-skewed-continuous variables (e.g.
    purely continuous-vital studies).
    """
    rules = variable_kind_constraints(
        context.variables,
        outcome_names=(
            context.target_outcome,
            *(context.cohort.outcome_columns or ()),
        ),
    )
    if not rules:
        return ""
    lines = [
        "Variable-type method-compatibility self-review checklist (derived",
        "automatically from the research context — must be honoured when",
        "drafting plans and writing code):",
    ]
    rendered_rules: List[Dict[str, object]]
    if compact:
        grouped: Dict[str, Dict[str, object]] = {}
        for rule in rules:
            kind = str(rule["kind"])
            entry = grouped.setdefault(
                kind,
                {
                    **rule,
                    "variables": [],
                },
            )
            variables = entry["variables"]
            assert isinstance(variables, list)
            variables.append(str(rule["variable"]))
        rendered_rules = list(grouped.values())
    else:
        rendered_rules = [
            {
                **rule,
                "variables": [str(rule["variable"])],
            }
            for rule in rules
        ]
    for r in rendered_rules:
        forbidden_head = ", ".join(list(r["forbidden_patterns"])[:5])  # type: ignore[index]
        preferred = ", ".join(list(r["preferred"]))  # type: ignore[index]
        variables = ", ".join(f"`{name}`" for name in r["variables"])  # type: ignore[union-attr]
        variable_label = (
            variables
            if len(r["variables"]) == 1  # type: ignore[arg-type]
            else f"variables {variables}"
        )
        lines.append(
            f"  - {variable_label} (kind: {r['kind']}): DO NOT use "
            f"{forbidden_head}. PREFERRED: {preferred}. "
            f"Reason: {r['rationale']}"
        )
    return "\n".join(lines)


def render_computational_budget_constraints(context: ResearchContext) -> str:
    """Publish deterministic bounds for quadratic metrics on large cohorts."""

    n_stays = int(context.cohort.n_stays)
    if n_stays <= PAIRWISE_EVALUATION_FULL_COHORT_MAX_ROWS:
        return ""
    return "\n".join(
        [
            "LARGE-COHORT COMPUTATIONAL BUDGET (host-owned; must be honoured):",
            f"- The execution cohort has {n_stays} stays. Preserve full-data model "
            "fitting and final label assignment.",
            "- Pairwise silhouette evaluation is quadratic: every "
            "`sklearn.metrics.silhouette_score` call must set "
            f"`sample_size <= {PAIRWISE_EVALUATION_MAX_SAMPLE_SIZE}` and an explicit "
            "deterministic `random_state`. This applies inside candidate and seed "
            "loops as well as to the final selected model.",
            "- Record the silhouette evaluation sample size and seed in the step "
            "summary or metric artifact so the approximation is replayable.",
        ]
    )


# ---------------------------------------------------------------------------
# Post-codegen pre-execution enforcement (Patch C)
# ---------------------------------------------------------------------------


import ast
import re


def _helper_call_contract_violations(code: str) -> List[Dict[str, object]]:
    """Return deterministic API-contract violations for documented helpers.

    This is intentionally limited to project-local method-suite helpers whose
    keyword-only signature prevents scientifically meaningful columns from
    being swapped accidentally.  Syntax-invalid scripts remain the syntax
    gate's responsibility.
    """

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    violations: List[Dict[str, object]] = []
    required_keywords = {
        "count_column",
        "measured_column",
        "representative_column",
    }
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function_name = None
        if isinstance(node.func, ast.Name):
            function_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            function_name = node.func.attr
        if function_name != "reconcile_binary_event_presence":
            continue

        keyword_names = {
            keyword.arg for keyword in node.keywords if keyword.arg is not None
        }
        has_expansion = any(keyword.arg is None for keyword in node.keywords)
        frame_is_bound = len(node.args) == 1 or (
            len(node.args) == 0 and "frame" in keyword_names
        )
        if (
            frame_is_bound
            and not has_expansion
            and len(node.args) <= 1
            and required_keywords <= keyword_names
        ):
            continue
        violations.append(
            {
                "variable": "reconcile_binary_event_presence",
                "kind": "method_helper_call_contract",
                "matched_patterns": ["positional_or_incomplete_sparse_event_call"],
                "preferred": (
                    "frame positional or named",
                    "count_column=...",
                    "measured_column=...",
                    "representative_column=...",
                ),
                "rationale": (
                    "The sparse-event columns are keyword-only so their clinical "
                    "roles cannot be silently swapped."
                ),
            }
        )
    return violations


def _static_integer_upper_bound(
    node: ast.AST,
    *,
    assignments: Dict[str, ast.AST],
    seen: Optional[set[str]] = None,
) -> Optional[int]:
    """Resolve a provable integer upper bound for a simple AST expression."""

    if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(
        node.value, bool
    ):
        return int(node.value)
    if isinstance(node, ast.Name):
        active_seen = set() if seen is None else set(seen)
        if node.id in active_seen or node.id not in assignments:
            return None
        active_seen.add(node.id)
        return _static_integer_upper_bound(
            assignments[node.id],
            assignments=assignments,
            seen=active_seen,
        )
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "int"
        and len(node.args) == 1
        and not node.keywords
    ):
        return _static_integer_upper_bound(
            node.args[0],
            assignments=assignments,
            seen=seen,
        )
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "min"
    ):
        bounds = [
            bound
            for argument in node.args
            if (
                bound := _static_integer_upper_bound(
                    argument,
                    assignments=assignments,
                    seen=seen,
                )
            )
            is not None
        ]
        return min(bounds) if bounds else None
    if isinstance(node, ast.IfExp):
        branch_bounds = [
            _static_integer_upper_bound(
                branch,
                assignments=assignments,
                seen=seen,
            )
            for branch in (node.body, node.orelse)
        ]
        if all(bound is not None for bound in branch_bounds):
            return max(bound for bound in branch_bounds if bound is not None)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left_bound = _static_integer_upper_bound(
            node.left,
            assignments=assignments,
            seen=seen,
        )
        right_bound = _static_integer_upper_bound(
            node.right,
            assignments=assignments,
            seen=seen,
        )
        if left_bound is not None and right_bound is not None:
            return left_bound + right_bound
    return None


def _simple_assignments(scope: ast.AST) -> Dict[str, List[ast.AST]]:
    values: Dict[str, List[ast.AST]] = {}
    for node in ast.walk(scope):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                values.setdefault(target.id, []).append(node.value)
    return values


def _fixed_parameter_at_every_call(
    tree: ast.AST,
    function: ast.FunctionDef,
    parameter: str,
    assignments: Dict[str, ast.AST],
) -> bool:
    parameters = [argument.arg for argument in function.args.args]
    if parameter not in parameters:
        return False
    position = parameters.index(parameter)
    actuals: List[ast.AST] = []
    for candidate in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
        if _callable_path(candidate.func, {}).rsplit(".", 1)[-1] != function.name:
            continue
        actual = next(
            (item.value for item in candidate.keywords if item.arg == parameter),
            candidate.args[position] if position < len(candidate.args) else None,
        )
        if actual is None:
            return False
        actuals.append(actual)
    return bool(actuals) and all(
        _static_integer_upper_bound(actual, assignments=assignments) is not None
        for actual in actuals
    )


def _statically_deterministic_integer(
    node: Optional[ast.AST],
    *,
    tree: ast.AST,
    function: Optional[ast.FunctionDef],
    assignments: Dict[str, ast.AST],
    fixed_seed_names: set[str],
    seen: Optional[set[str]] = None,
) -> bool:
    """Return whether an integer expression is fixed by code and call sites."""

    if node is None:
        return False
    if _static_integer_upper_bound(node, assignments=assignments) is not None:
        return True
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "int"
        and len(node.args) == 1
        and not node.keywords
    ):
        return _statically_deterministic_integer(
            node.args[0],
            tree=tree,
            function=function,
            assignments=assignments,
            fixed_seed_names=fixed_seed_names,
            seen=seen,
        )
    if isinstance(node, ast.Name):
        if node.id in fixed_seed_names:
            return True
        if function is not None and node.id in {
            argument.arg for argument in function.args.args
        }:
            return _fixed_parameter_at_every_call(
                tree,
                function,
                node.id,
                assignments,
            )
        active_seen = set() if seen is None else set(seen)
        if node.id in active_seen or node.id not in assignments:
            return False
        active_seen.add(node.id)
        return _statically_deterministic_integer(
            assignments[node.id],
            tree=tree,
            function=function,
            assignments=assignments,
            fixed_seed_names=fixed_seed_names,
            seen=active_seen,
        )
    if isinstance(node, ast.IfExp):
        return all(
            _statically_deterministic_integer(
                branch,
                tree=tree,
                function=function,
                assignments=assignments,
                fixed_seed_names=fixed_seed_names,
                seen=seen,
            )
            for branch in (node.body, node.orelse)
        )
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return all(
            _statically_deterministic_integer(
                operand,
                tree=tree,
                function=function,
                assignments=assignments,
                fixed_seed_names=fixed_seed_names,
                seen=seen,
            )
            for operand in (node.left, node.right)
        )
    return False


def _permutation_sample_bounded_silhouette_contract(
    call: ast.Call,
    *,
    tree: ast.AST,
    function: Optional[ast.FunctionDef],
    assignments: Dict[str, ast.AST],
    fixed_seed_names: set[str],
) -> bool:
    """Accept a closed, label-preserving permutation sampler.

    Some repairs cannot pass ``sample_size``/``random_state`` to sklearn: they
    first materialise a deterministic sample, force one row per label, and
    then call the metric on those sampled arrays.  This recognizer accepts only
    that explicit shape.  It does not infer a bound from a comment or from a
    generic ``rng`` call; the sample cap and seed must be fixed at every helper
    call site, and the sampler must prove that every label survives sampling.
    """

    if function is None or len(call.args) < 2:
        return False

    local = _simple_assignments(function)
    parameters = [argument.arg for argument in function.args.args]
    if len(parameters) < 4:
        return False

    def _single_assignment(name: str) -> Optional[ast.AST]:
        values = local.get(name, [])
        return values[0] if len(values) == 1 else None

    def _name_call(node: Optional[ast.AST], path: str) -> bool:
        return (
            isinstance(node, ast.Call)
            and _callable_path(node.func, {}).casefold() == path.casefold()
        )

    labels_assignment = next(
        (
            (name, value)
            for name, values in local.items()
            for value in values
            if _name_call(value, "np.asarray")
            and len(value.args) == 1
            and isinstance(value.args[0], ast.Name)
            and value.args[0].id == parameters[1]
        ),
        None,
    )
    if labels_assignment is None:
        return False
    labels_name, _ = labels_assignment

    unique_labels_assignment = next(
        (
            (name, value)
            for name, values in local.items()
            for value in values
            if _name_call(value, "np.unique")
            and len(value.args) == 1
            and isinstance(value.args[0], ast.Name)
            and value.args[0].id == labels_name
        ),
        None,
    )
    if unique_labels_assignment is None:
        return False
    unique_labels_name, _ = unique_labels_assignment

    n_rows_assignment = next(
        (
            (name, value)
            for name, values in local.items()
            for value in values
            if isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id == "len"
            and len(value.args) == 1
            and isinstance(value.args[0], ast.Name)
            and value.args[0].id == labels_name
        ),
        None,
    )
    if n_rows_assignment is None:
        return False
    n_rows_name, _ = n_rows_assignment

    sample_size_assignment = next(
        (
            (name, value)
            for name, values in local.items()
            for value in values
            if isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id == "min"
            and len(value.args) == 2
            and isinstance(value.args[1], ast.Name)
            and value.args[1].id == n_rows_name
            and (
                (
                    isinstance(value.args[0], ast.Name)
                    and value.args[0].id in parameters
                )
                or (
                    isinstance(value.args[0], ast.Call)
                    and isinstance(value.args[0].func, ast.Name)
                    and value.args[0].func.id == "int"
                    and len(value.args[0].args) == 1
                    and isinstance(value.args[0].args[0], ast.Name)
                    and value.args[0].args[0].id in parameters
                )
            )
        ),
        None,
    )
    if sample_size_assignment is None:
        return False
    sample_size_name, sample_size_value = sample_size_assignment
    sample_cap_expression = sample_size_value.args[0]
    if isinstance(sample_cap_expression, ast.Call):
        sample_cap_parameter = sample_cap_expression.args[0].id
    else:
        sample_cap_parameter = sample_cap_expression.id

    rng_assignment = next(
        (
            (name, value)
            for name, values in local.items()
            for value in values
            if _name_call(value, "np.random.default_rng")
            and len(value.args) == 1
            and isinstance(value.args[0], ast.Name)
            and value.args[0].id in parameters
        ),
        None,
    )
    if rng_assignment is None:
        return False
    rng_name, rng_value = rng_assignment
    seed_parameter = rng_value.args[0].id

    permutation_assignment = next(
        (
            (name, value)
            for name, values in local.items()
            for value in values
            if isinstance(value, ast.Call)
            and isinstance(value.func, ast.Attribute)
            and isinstance(value.func.value, ast.Name)
            and value.func.value.id == rng_name
            and value.func.attr == "permutation"
            and len(value.args) == 1
            and isinstance(value.args[0], ast.Name)
            and value.args[0].id == n_rows_name
        ),
        None,
    )
    if permutation_assignment is None:
        return False
    permutation_name, _ = permutation_assignment

    selected_name = next(
        (
            name
            for name, values in local.items()
            if len(values) == 1 and isinstance(values[0], ast.List)
        ),
        None,
    )
    selected_set_name = next(
        (
            name
            for name, values in local.items()
            if len(values) == 1
            and isinstance(values[0], ast.Call)
            and isinstance(values[0].func, ast.Name)
            and values[0].func.id == "set"
            and not values[0].args
        ),
        None,
    )
    if selected_name is None or selected_set_name is None:
        return False

    def _contains_first_label_selection(loop: ast.For) -> bool:
        if not (
            isinstance(loop.iter, ast.Name)
            and loop.iter.id == unique_labels_name
        ):
            return False
        has_cluster_indices = False
        has_append = False
        has_add = False
        for node in ast.walk(loop):
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and isinstance(node.value, ast.Call)
                and _callable_path(node.value.func, {}).rsplit(".", 1)[-1]
                == "flatnonzero"
            ):
                has_cluster_indices = True
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == selected_name
                and node.func.attr == "append"
            ):
                has_append = True
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == selected_set_name
                and node.func.attr == "add"
            ):
                has_add = True
        return has_cluster_indices and has_append and has_add

    def _contains_permutation_fill(loop: ast.For) -> bool:
        if not (
            isinstance(loop.target, ast.Name)
            and isinstance(loop.iter, ast.Name)
            and loop.iter.id == permutation_name
        ):
            return False
        has_membership_guard = False
        has_append = False
        has_add = False
        has_cap_break = False
        for node in ast.walk(loop):
            if (
                isinstance(node, ast.Compare)
                and len(node.ops) == 1
                and isinstance(node.ops[0], ast.NotIn)
                and isinstance(node.left, ast.Name)
                and node.left.id == loop.target.id
                and len(node.comparators) == 1
                and isinstance(node.comparators[0], ast.Name)
                and node.comparators[0].id == selected_set_name
            ):
                has_membership_guard = True
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == selected_name
                and node.func.attr == "append"
            ):
                has_append = True
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == selected_set_name
                and node.func.attr == "add"
            ):
                has_add = True
            if (
                isinstance(node, ast.Compare)
                and len(node.ops) == 1
                and isinstance(node.ops[0], ast.Eq)
                and isinstance(node.left, ast.Call)
                and isinstance(node.left.func, ast.Name)
                and node.left.func.id == "len"
                and len(node.left.args) == 1
                and isinstance(node.left.args[0], ast.Name)
                and node.left.args[0].id == selected_name
                and len(node.comparators) == 1
                and isinstance(node.comparators[0], ast.Name)
                and node.comparators[0].id == sample_size_name
            ):
                has_cap_break = any(isinstance(child, ast.Break) for child in ast.walk(loop))
        return has_membership_guard and has_append and has_add and has_cap_break

    loops = [node for node in ast.walk(function) if isinstance(node, ast.For)]
    if not any(_contains_first_label_selection(loop) for loop in loops):
        return False
    if not any(_contains_permutation_fill(loop) for loop in loops):
        return False

    sample_indices_assignment = next(
        (
            (name, value)
            for name, values in local.items()
            for value in values
            if _name_call(value, "np.asarray")
            and len(value.args) == 1
            and isinstance(value.args[0], ast.Name)
            and value.args[0].id == selected_name
        ),
        None,
    )
    if sample_indices_assignment is None:
        return False
    sample_indices_name, _ = sample_indices_assignment
    sample_labels_assignment = next(
        (
            (name, value)
            for name, values in local.items()
            for value in values
            if isinstance(value, ast.Subscript)
            and isinstance(value.value, ast.Name)
            and value.value.id == labels_name
            and isinstance(value.slice, ast.Name)
            and value.slice.id == sample_indices_name
        ),
        None,
    )
    if sample_labels_assignment is None:
        return False
    sample_labels_name, _ = sample_labels_assignment

    if not any(
        isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and len(node.test.ops) == 1
        and isinstance(node.test.ops[0], ast.NotEq)
        and isinstance(node.test.left, ast.Attribute)
        and node.test.left.attr == "size"
        and isinstance(node.test.left.value, ast.Call)
        and _callable_path(node.test.left.value.func, {}).rsplit(".", 1)[-1]
        == "unique"
        and node.test.left.value.args
        and isinstance(node.test.left.value.args[0], ast.Name)
        and node.test.left.value.args[0].id == sample_labels_name
        and len(node.test.comparators) == 1
        and isinstance(node.test.comparators[0], ast.Attribute)
        and node.test.comparators[0].attr == "size"
        and isinstance(node.test.comparators[0].value, ast.Name)
        and node.test.comparators[0].value.id == unique_labels_name
        for node in ast.walk(function)
    ):
        return False

    if not (
        isinstance(call.args[0], ast.Subscript)
        and isinstance(call.args[0].slice, ast.Name)
        and call.args[0].slice.id == sample_indices_name
        and isinstance(call.args[1], ast.Name)
        and call.args[1].id == sample_labels_name
    ):
        return False

    def _call_site_actuals(parameter: str) -> Optional[List[ast.AST]]:
        position = parameters.index(parameter)
        actuals: List[ast.AST] = []
        for candidate in ast.walk(tree):
            if not isinstance(candidate, ast.Call):
                continue
            if _callable_path(candidate.func, {}).rsplit(".", 1)[-1] != function.name:
                continue
            actual = next(
                (item.value for item in candidate.keywords if item.arg == parameter),
                candidate.args[position] if position < len(candidate.args) else None,
            )
            if actual is None:
                return None
            actuals.append(actual)
        return actuals or None

    cap_actuals = _call_site_actuals(sample_cap_parameter)
    seed_actuals = _call_site_actuals(seed_parameter)
    if cap_actuals is None or seed_actuals is None:
        return False
    if any(
        (bound := _static_integer_upper_bound(actual, assignments=assignments)) is None
        or bound > PAIRWISE_EVALUATION_MAX_SAMPLE_SIZE
        for actual in cap_actuals
    ):
        return False
    return all(
        _statically_deterministic_integer(
            actual,
            tree=tree,
            function=None,
            assignments=assignments,
            fixed_seed_names=fixed_seed_names,
        )
        for actual in seed_actuals
    )


def _choice_sample_bounded_silhouette_contract(
    call: ast.Call,
    *,
    tree: ast.AST,
    function: Optional[ast.FunctionDef],
    assignments: Dict[str, ast.AST],
    fixed_seed_names: set[str],
) -> bool:
    """Accept a deterministic ``Generator.choice`` silhouette sampler.

    A generated step may use the direct, bounded shape::

        sample_n = min(int(max_sample_size), int(matrix.shape[0]))
        rng = np.random.default_rng(int(seed))
        indices = rng.choice(matrix.shape[0], size=sample_n, replace=False)
        return silhouette_score(matrix[indices], labels[indices])

    The sklearn call has no ``sample_size``/``random_state`` keywords in this
    form, so the recognizer must prove the bound and seed at the helper's
    definition and at every call site.  It deliberately does not accept a
    generic RNG call, a full-data overwrite, or a dynamic seed.
    """

    if function is None or len(call.args) < 2:
        return False

    parameters = [argument.arg for argument in function.args.args]
    if len(parameters) < 3:
        return False
    local = _simple_assignments(function)

    def _name_call(node: Optional[ast.AST], path: str) -> bool:
        return (
            isinstance(node, ast.Call)
            and _callable_path(node.func, {}).casefold() == path.casefold()
        )

    def _subscript_index(node: ast.AST) -> Optional[str]:
        if not isinstance(node, ast.Subscript):
            return None
        index = node.slice
        if isinstance(index, ast.Tuple) and index.elts:
            index = index.elts[0]
        return index.id if isinstance(index, ast.Name) else None

    def _same_population(left: ast.AST, right: ast.AST) -> bool:
        def _unwrap_int(node: ast.AST) -> ast.AST:
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "int"
                and len(node.args) == 1
                and not node.keywords
            ):
                return _unwrap_int(node.args[0])
            return node

        left = _unwrap_int(left)
        right = _unwrap_int(right)
        return ast.dump(left, include_attributes=False) == ast.dump(
            right, include_attributes=False
        )

    sample_assignment: Optional[tuple[str, ast.Call, ast.AST, ast.AST]] = None
    for name, values in local.items():
        if len(values) != 1 or not isinstance(values[0], ast.Call):
            continue
        value = values[0]
        if not (
            isinstance(value.func, ast.Name)
            and value.func.id == "min"
            and len(value.args) == 2
            and not value.keywords
        ):
            continue
        cap_expr, population_expr = value.args
        if isinstance(cap_expr, ast.Call) and (
            not isinstance(cap_expr.func, ast.Name)
            or cap_expr.func.id != "int"
            or len(cap_expr.args) != 1
            or cap_expr.keywords
        ):
            continue
        if isinstance(cap_expr, ast.Call):
            cap_expr = cap_expr.args[0]
        sample_assignment = (name, value, cap_expr, population_expr)
        break
    if sample_assignment is None:
        return False
    sample_name, _, cap_expr, population_expr = sample_assignment

    cap_parameter = (
        cap_expr.id if isinstance(cap_expr, ast.Name) and cap_expr.id in parameters else None
    )
    if cap_parameter is None:
        return False

    # The population expression may be ``n_rows`` or ``matrix.shape[0]``;
    # either is safe as long as the choice call uses that same expression.
    rng_name: Optional[str] = None
    seed_expr: Optional[ast.AST] = None
    for name, values in local.items():
        if len(values) != 1 or not _name_call(values[0], "np.random.default_rng"):
            continue
        rng_call = values[0]
        if len(rng_call.args) != 1 or rng_call.keywords:
            continue
        rng_name = name
        seed_expr = rng_call.args[0]
        if (
            isinstance(seed_expr, ast.Call)
            and isinstance(seed_expr.func, ast.Name)
            and seed_expr.func.id == "int"
            and len(seed_expr.args) == 1
            and not seed_expr.keywords
        ):
            seed_expr = seed_expr.args[0]
        break
    if rng_name is None or seed_expr is None:
        return False

    choice_name: Optional[str] = None
    choice_call: Optional[ast.Call] = None
    choice_assignment: Optional[ast.Assign] = None
    for name, values in local.items():
        for value in values:
            if not isinstance(value, ast.Call):
                continue
            candidate = value
            if (
                isinstance(candidate.func, ast.Attribute)
                and candidate.func.attr == "sort"
                and len(candidate.args) == 1
                and not candidate.keywords
            ):
                candidate = candidate.args[0]
            if not (
                isinstance(candidate.func, ast.Attribute)
                and isinstance(candidate.func.value, ast.Name)
                and candidate.func.value.id == rng_name
                and candidate.func.attr == "choice"
                and len(candidate.args) == 1
            ):
                continue
            keywords = {item.arg: item.value for item in candidate.keywords if item.arg}
            if (
                not _same_population(candidate.args[0], population_expr)
                or not isinstance(keywords.get("size"), ast.Name)
                or keywords["size"].id != sample_name
                or not isinstance(keywords.get("replace"), ast.Constant)
                or keywords["replace"].value is not False
                or len(keywords) != 2
            ):
                continue
            choice_name = name
            choice_call = candidate
            choice_assignment = next(
                (
                    node
                    for node in ast.walk(function)
                    if isinstance(node, ast.Assign)
                    and any(
                        isinstance(target, ast.Name) and target.id == name
                        for target in node.targets
                    )
                    and (
                        node.value is value
                        or (
                            isinstance(node.value, ast.Call)
                            and isinstance(node.value.func, ast.Attribute)
                            and node.value.func.attr == "sort"
                            and len(node.value.args) == 1
                            and node.value.args[0] is candidate
                        )
                    )
                ),
                None,
            )
            break
        if choice_name is not None:
            break
    if choice_name is None or choice_call is None or choice_assignment is None:
        return False

    # Require the helper to pass the sampled indices into the metric.  A
    # direct subscript is preferred, but a local ``sampled_features`` /
    # ``sampled_labels`` assignment is also safe when it is not overwritten.
    index_name = choice_name
    metric_feature_index = _subscript_index(call.args[0])
    metric_label_index = _subscript_index(call.args[1])
    if metric_feature_index != index_name:
        feature_name = call.args[0].value.id if (
            isinstance(call.args[0], ast.Subscript)
            and isinstance(call.args[0].value, ast.Name)
        ) else None
        if feature_name is None:
            return False
        feature_values = local.get(feature_name, [])
        if len(feature_values) != 1 or _subscript_index(feature_values[0]) != index_name:
            return False
    if metric_label_index != index_name:
        if not isinstance(call.args[1], ast.Name):
            return False
        label_values = local.get(call.args[1].id, [])
        if not any(_subscript_index(value) == index_name for value in label_values):
            return False

    # The helper must have a fixed-size branch (or an equivalent guard) so the
    # full cohort is never passed to the metric when the cohort is large.
    has_bounded_branch: Optional[ast.If] = None
    for node in ast.walk(function):
        if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
            continue
        if not (
            len(node.test.ops) == 1
            and isinstance(node.test.ops[0], ast.Lt)
            and isinstance(node.test.left, ast.Name)
            and node.test.left.id == sample_name
            and len(node.test.comparators) == 1
            and _same_population(node.test.comparators[0], population_expr)
            and node.orelse
        ):
            continue
        has_bounded_branch = node
        break
    if has_bounded_branch is None:
        return False

    def _is_descendant(node: ast.AST, ancestor: ast.AST) -> bool:
        return any(candidate is node for candidate in ast.walk(ancestor))

    if not any(
        _is_descendant(choice_assignment, statement)
        for statement in has_bounded_branch.body
    ):
        return False
    if getattr(call, "lineno", 0) <= getattr(has_bounded_branch, "end_lineno", 0):
        return False
    if getattr(choice_assignment, "lineno", 0) >= getattr(call, "lineno", 0):
        return False
    # An index assignment after the bounded branch is an unsafe full-cohort
    # overwrite. The only permitted second assignment is the mutually
    # exclusive full-index fallback in the branch's ``else`` arm.
    for node in ast.walk(function):
        if not (
            isinstance(node, ast.Assign)
            and getattr(node, "lineno", 0) < getattr(call, "lineno", 0)
            and any(
                isinstance(target, ast.Name) and target.id == index_name
                for target in node.targets
            )
        ):
            continue
        if node is choice_assignment:
            continue
        if not any(
            _is_descendant(node, statement)
            for statement in has_bounded_branch.orelse
        ):
            return False

    def _call_site_actuals(parameter: str) -> Optional[List[ast.AST]]:
        position = parameters.index(parameter)
        defaults_start = len(parameters) - len(function.args.defaults)
        default = (
            function.args.defaults[position - defaults_start]
            if position >= defaults_start
            else None
        )
        actuals: List[ast.AST] = []
        for candidate in ast.walk(tree):
            if not isinstance(candidate, ast.Call):
                continue
            if _callable_path(candidate.func, {}).rsplit(".", 1)[-1] != function.name:
                continue
            actual = next(
                (item.value for item in candidate.keywords if item.arg == parameter),
                candidate.args[position] if position < len(candidate.args) else default,
            )
            if actual is None:
                return None
            actuals.append(actual)
        return actuals or None

    cap_actuals = _call_site_actuals(cap_parameter)
    if cap_actuals is None or any(
        (bound := _static_integer_upper_bound(actual, assignments=assignments)) is None
        or bound > PAIRWISE_EVALUATION_MAX_SAMPLE_SIZE
        for actual in cap_actuals
    ):
        return False
    return all(
        _statically_deterministic_integer(
            actual,
            tree=tree,
            function=None,
            assignments=assignments,
            fixed_seed_names=fixed_seed_names,
        )
        for actual in _call_site_actuals(
            next(
                (parameter for parameter in parameters if parameter == seed_expr.id),
                parameters[2],
            )
        )
        or []
    )


def _manual_bounded_silhouette_contract(
    call: ast.Call,
    *,
    tree: ast.AST,
    function: Optional[ast.FunctionDef],
    assignments: Dict[str, ast.AST],
    fixed_seed_names: set[str],
) -> bool:
    """Accept the explicit bounded-subsample shape observed in generated code."""

    if function is None or len(call.args) < 2:
        return False

    if _permutation_sample_bounded_silhouette_contract(
        call,
        tree=tree,
        function=function,
        assignments=assignments,
        fixed_seed_names=fixed_seed_names,
    ):
        return True

    if _choice_sample_bounded_silhouette_contract(
        call,
        tree=tree,
        function=function,
        assignments=assignments,
        fixed_seed_names=fixed_seed_names,
    ):
        return True

    # A repair may preserve the original sklearn callable under an alias and
    # bind its required budget arguments through ``**kwargs``. Accept only the
    # closed wrapper shape we can prove: two unconditional subscript writes,
    # followed immediately by the aliased call. Conditional/default-dependent
    # updates and any later mutation remain untrusted.
    kwarg = function.args.kwarg
    if kwarg is not None and len(function.body) == 3:
        kwarg_name = kwarg.arg
        writes: Dict[str, ast.AST] = {}
        writes_are_closed = True
        for statement in function.body[:2]:
            if not (
                isinstance(statement, ast.Assign)
                and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Subscript)
                and isinstance(statement.targets[0].value, ast.Name)
                and statement.targets[0].value.id == kwarg_name
                and isinstance(statement.targets[0].slice, ast.Constant)
                and isinstance(statement.targets[0].slice.value, str)
                and statement.targets[0].slice.value not in writes
            ):
                writes_are_closed = False
                break
            writes[statement.targets[0].slice.value] = statement.value
        positional_parameters = [argument.arg for argument in function.args.args]
        return_statement = function.body[-1]
        forwards_kwargs = (
            isinstance(return_statement, ast.Return)
            and return_statement.value is call
            and len(call.keywords) == 1
            and call.keywords[0].arg is None
            and isinstance(call.keywords[0].value, ast.Name)
            and call.keywords[0].value.id == kwarg_name
            and len(positional_parameters) >= 2
            and len(call.args) >= 2
            and all(
                isinstance(actual, ast.Name)
                and actual.id == positional_parameters[index]
                for index, actual in enumerate(call.args[:2])
            )
        )
        sample_bound = _static_integer_upper_bound(
            writes.get("sample_size"),
            assignments=assignments,
        )
        deterministic_seed = _statically_deterministic_integer(
            writes.get("random_state"),
            tree=tree,
            function=function,
            assignments=assignments,
            fixed_seed_names=fixed_seed_names,
        )
        if (
            writes_are_closed
            and set(writes) == {"sample_size", "random_state"}
            and forwards_kwargs
            and sample_bound is not None
            and sample_bound <= PAIRWISE_EVALUATION_MAX_SAMPLE_SIZE
            and deterministic_seed
        ):
            return True

    local = _simple_assignments(function)

    def indexed_by(node: ast.AST) -> Optional[str]:
        candidate = node
        if isinstance(candidate, ast.Name):
            values = local.get(candidate.id, [])
            if len(values) != 1:
                return None
            candidate = values[0]
        if not isinstance(candidate, ast.Subscript):
            return None
        index = candidate.slice
        if isinstance(index, ast.Tuple) and index.elts:
            index = index.elts[0]
        return index.id if isinstance(index, ast.Name) else None

    index_names = {indexed_by(item) for item in call.args[:2]}
    if len(index_names) == 1 and None not in index_names:
        index_name = next(iter(index_names))
        parameters = [argument.arg for argument in function.args.args]

        def parameter_actuals(parameter: str) -> Optional[List[ast.AST]]:
            if parameter not in parameters:
                return None
            position = parameters.index(parameter)
            actuals: List[ast.AST] = []
            for candidate in (
                node for node in ast.walk(tree) if isinstance(node, ast.Call)
            ):
                if (
                    _callable_path(candidate.func, {}).rsplit(".", 1)[-1]
                    != function.name
                ):
                    continue
                actual = next(
                    (
                        item.value
                        for item in candidate.keywords
                        if item.arg == parameter
                    ),
                    (
                        candidate.args[position]
                        if position < len(candidate.args)
                        else None
                    ),
                )
                if actual is None:
                    return None
                actuals.append(actual)
            return actuals or None

        for branch in (
            item for item in function.body if isinstance(item, ast.If)
        ):
            test = branch.test
            if not (
                isinstance(test, ast.Compare)
                and len(test.ops) == len(test.comparators) == 1
                and isinstance(test.ops[0], ast.Eq)
                and isinstance(test.left, ast.Name)
                and isinstance(test.comparators[0], ast.Name)
                and branch.orelse
            ):
                continue
            sample_name = test.left.id
            population_name = test.comparators[0].id
            sample_parameter: Optional[str] = None
            for value in local.get(sample_name, []):
                if not (
                    isinstance(value, ast.Call)
                    and isinstance(value.func, ast.Name)
                    and value.func.id == "min"
                    and any(
                        isinstance(argument, ast.Name)
                        and argument.id == population_name
                        for argument in value.args
                    )
                ):
                    continue
                for argument in value.args:
                    candidate = argument
                    if (
                        isinstance(candidate, ast.Call)
                        and isinstance(candidate.func, ast.Name)
                        and candidate.func.id == "int"
                        and len(candidate.args) == 1
                    ):
                        candidate = candidate.args[0]
                    if (
                        isinstance(candidate, ast.Name)
                        and candidate.id != population_name
                        and candidate.id in parameters
                    ):
                        sample_parameter = candidate.id
                        break
            if sample_parameter is None:
                continue
            sample_actuals = parameter_actuals(sample_parameter)
            if sample_actuals is None or any(
                (bound := _static_integer_upper_bound(
                    actual,
                    assignments=assignments,
                ))
                is None
                or bound > PAIRWISE_EVALUATION_MAX_SAMPLE_SIZE
                for actual in sample_actuals
            ):
                continue

            body = _simple_assignments(ast.Module(body=branch.body, type_ignores=[]))
            otherwise = _simple_assignments(
                ast.Module(body=branch.orelse, type_ignores=[])
            )
            branch_values = body.get(index_name, [])
            otherwise_values = otherwise.get(index_name, [])
            if len(branch_values) != 1 or len(otherwise_values) != 1:
                continue

            def is_full_index(node: ast.AST) -> bool:
                return bool(
                    isinstance(node, ast.Call)
                    and _callable_path(node.func, {}).rsplit(".", 1)[-1]
                    == "arange"
                    and len(node.args) == 1
                    and isinstance(node.args[0], ast.Name)
                    and node.args[0].id == population_name
                )

            def choice_call(node: ast.AST) -> Optional[ast.Call]:
                candidate = node
                if (
                    isinstance(candidate, ast.Call)
                    and _callable_path(candidate.func, {}).rsplit(".", 1)[-1]
                    == "sort"
                    and len(candidate.args) == 1
                ):
                    candidate = candidate.args[0]
                return (
                    candidate
                    if isinstance(candidate, ast.Call)
                    and _callable_path(candidate.func, {}).rsplit(".", 1)[-1]
                    == "choice"
                    else None
                )

            full_value, sampled_value = branch_values[0], otherwise_values[0]
            if not is_full_index(full_value):
                full_value, sampled_value = sampled_value, full_value
            choice = choice_call(sampled_value)
            if not is_full_index(full_value) or choice is None:
                continue
            keywords = {
                item.arg: item.value for item in choice.keywords if item.arg
            }
            if not (
                choice.args
                and isinstance(choice.args[0], ast.Name)
                and choice.args[0].id == population_name
                and isinstance(keywords.get("size"), ast.Name)
                and keywords["size"].id == sample_name
                and isinstance(keywords.get("replace"), ast.Constant)
                and keywords["replace"].value is False
                and isinstance(choice.func, ast.Attribute)
                and isinstance(choice.func.value, ast.Name)
            ):
                continue
            rng_name = choice.func.value.id
            rng_values = local.get(rng_name, [])
            if len(rng_values) != 1:
                continue
            rng = rng_values[0]
            if not (
                isinstance(rng, ast.Call)
                and _callable_path(rng.func, {}).rsplit(".", 1)[-1]
                == "default_rng"
                and len(rng.args) == 1
            ):
                continue
            seed = rng.args[0]
            if (
                isinstance(seed, ast.Call)
                and isinstance(seed.func, ast.Name)
                and seed.func.id == "int"
                and len(seed.args) == 1
            ):
                seed = seed.args[0]
            if not isinstance(seed, ast.Name):
                continue
            seed_actuals = parameter_actuals(seed.id)
            if seed_actuals and all(
                _statically_deterministic_integer(
                    actual,
                    tree=tree,
                    function=None,
                    assignments=assignments,
                    fixed_seed_names=fixed_seed_names,
                )
                for actual in seed_actuals
            ):
                return True

    if not all(isinstance(item, ast.Name) for item in call.args[:2]):
        return False
    call_inputs = [item.id for item in call.args[:2] if isinstance(item, ast.Name)]
    parameters = {argument.arg for argument in function.args.args}

    for branch in (item for item in function.body if isinstance(item, ast.If)):
        test = branch.test
        if not (
            isinstance(test, ast.Compare)
            and isinstance(test.left, ast.Name)
            and len(test.ops) == len(test.comparators) == 1
            and isinstance(test.ops[0], ast.Lt)
            and isinstance(test.comparators[0], ast.Name)
            and branch.orelse
        ):
            continue
        sample_name = test.left.id
        population_name = test.comparators[0].id
        sample_bound = next(
            (
                _static_integer_upper_bound(value, assignments=assignments)
                for value in local.get(sample_name, [])
                if isinstance(value, ast.Call)
                and isinstance(value.func, ast.Name)
                and value.func.id == "min"
                and any(
                    isinstance(argument, ast.Name)
                    and argument.id == population_name
                    for argument in value.args
                )
            ),
            None,
        )
        if sample_bound is None or sample_bound > PAIRWISE_EVALUATION_MAX_SAMPLE_SIZE:
            continue

        body = _simple_assignments(ast.Module(body=branch.body, type_ignores=[]))
        otherwise = _simple_assignments(ast.Module(body=branch.orelse, type_ignores=[]))
        index_names = []
        for input_name in call_inputs:
            values = body.get(input_name, [])
            if len(local.get(input_name, [])) != 2 or input_name not in otherwise:
                break
            index_names.append(
                {
                    value.slice.id
                    for value in values
                    if isinstance(value, ast.Subscript)
                    and isinstance(value.slice, ast.Name)
                }
            )
        else:
            common_indices = set.intersection(*index_names)
            for index_name in common_indices:
                for choice in body.get(index_name, []):
                    if not isinstance(choice, ast.Call):
                        continue
                    keywords = {
                        item.arg: item.value for item in choice.keywords if item.arg
                    }
                    if (
                        _callable_path(choice.func, {}).rsplit(".", 1)[-1] != "choice"
                        or _static_integer_upper_bound(
                            keywords.get("size"), assignments=assignments
                        )
                        is None
                        or not isinstance(keywords.get("replace"), ast.Constant)
                        or keywords["replace"].value is not False
                        or not isinstance(choice.func, ast.Attribute)
                        or not isinstance(choice.func.value, ast.Name)
                    ):
                        continue
                    for rng in local.get(choice.func.value.id, []):
                        if not (
                            isinstance(rng, ast.Call)
                            and _callable_path(rng.func, {}).rsplit(".", 1)[-1]
                            == "default_rng"
                            and len(rng.args) == 1
                        ):
                            continue
                        seed = rng.args[0]
                        if (
                            isinstance(seed, ast.Call)
                            and isinstance(seed.func, ast.Name)
                            and seed.func.id == "int"
                            and len(seed.args) == 1
                        ):
                            seed = seed.args[0]
                        if _static_integer_upper_bound(seed, assignments=assignments) is not None:
                            return True
                        if (
                            isinstance(seed, ast.Name)
                            and seed.id in parameters
                            and _fixed_parameter_at_every_call(
                                tree, function, seed.id, assignments
                            )
                        ):
                            return True
    return False


def _large_cohort_silhouette_violations(
    code: str,
    *,
    n_stays: int,
) -> List[Dict[str, object]]:
    """Reject unbounded quadratic silhouette calls for a large cohort."""

    if n_stays <= PAIRWISE_EVALUATION_FULL_COHORT_MAX_ROWS:
        return []
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    assignments: Dict[str, ast.AST] = {}
    imported_names: set[str] = set()
    local_function_names: set[str] = set()
    fixed_seed_names: set[str] = set()
    enclosing_functions: Dict[int, ast.FunctionDef] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            local_function_names.add(node.name.casefold())
            for descendant in ast.walk(node):
                enclosing_functions.setdefault(id(descendant), node)
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.NamedExpr)):
            if isinstance(node, ast.Assign):
                targets = node.targets
                value = node.value
            else:
                targets = [node.target]
                value = node.value
            if value is not None:
                for target in targets:
                    if isinstance(target, ast.Name):
                        assignments[target.id] = value
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name == "silhouette_score":
                    imported_names.add(alias.asname or alias.name)
        elif (
            isinstance(node, ast.For)
            and isinstance(node.target, ast.Name)
            and (
                (
                    isinstance(node.iter, (ast.List, ast.Set, ast.Tuple))
                    and node.iter.elts
                    and all(
                        isinstance(item, ast.Constant)
                        and isinstance(item.value, int)
                        and not isinstance(item.value, bool)
                        for item in node.iter.elts
                    )
                )
                or (
                    isinstance(node.iter, ast.Name)
                    and isinstance(assignments.get(node.iter.id), (ast.List, ast.Set, ast.Tuple))
                    and assignments[node.iter.id].elts
                    and all(
                        isinstance(item, ast.Constant)
                        and isinstance(item.value, int)
                        and not isinstance(item.value, bool)
                        for item in assignments[node.iter.id].elts
                    )
                )
                or (
                    isinstance(node.iter, ast.Call)
                    and isinstance(node.iter.func, ast.Name)
                    and node.iter.func.id == "range"
                    and 1 <= len(node.iter.args) <= 3
                    and not node.iter.keywords
                    and all(
                        _static_integer_upper_bound(
                            argument,
                            assignments=assignments,
                        )
                        is not None
                        for argument in node.iter.args
                    )
                )
            )
        ):
            fixed_seed_names.add(node.target.id)

    violations: List[Dict[str, object]] = []
    imported_names_casefold = {name.casefold() for name in imported_names}
    for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
        callable_path = _callable_path(call.func, {})
        terminal_name = callable_path.rsplit(".", 1)[-1]
        direct_import = (
            isinstance(call.func, ast.Name)
            and terminal_name in imported_names_casefold
        )
        attribute_call = (
            isinstance(call.func, ast.Attribute)
            and terminal_name == "silhouette_score"
        )
        unresolved_bare_call = (
            isinstance(call.func, ast.Name)
            and terminal_name == "silhouette_score"
            and terminal_name not in local_function_names
        )
        if not (direct_import or attribute_call or unresolved_bare_call):
            continue
        keywords = {
            keyword.arg: keyword.value
            for keyword in call.keywords
            if keyword.arg is not None
        }
        sample_bound = (
            _static_integer_upper_bound(
                keywords["sample_size"],
                assignments=assignments,
            )
            if "sample_size" in keywords
            else None
        )
        missing_contracts: List[str] = []
        if (
            sample_bound is None
            or sample_bound > PAIRWISE_EVALUATION_MAX_SAMPLE_SIZE
        ):
            missing_contracts.append("sample_size")
        deterministic_random_state = _statically_deterministic_integer(
            keywords.get("random_state"),
            tree=tree,
            function=enclosing_functions.get(id(call)),
            assignments=assignments,
            fixed_seed_names=fixed_seed_names,
        )
        manual_bounded_contract = _manual_bounded_silhouette_contract(
            call,
            tree=tree,
            function=enclosing_functions.get(id(call)),
            assignments=assignments,
            fixed_seed_names=fixed_seed_names,
        )
        if manual_bounded_contract:
            sample_bound = PAIRWISE_EVALUATION_MAX_SAMPLE_SIZE
            deterministic_random_state = True
            missing_contracts = []
        if not deterministic_random_state:
            missing_contracts.append("random_state")
        if not missing_contracts:
            continue
        violations.append(
            {
                "reason_code": "large_cohort_silhouette_unbounded",
                "variable": "silhouette_score",
                "kind": "computational_budget",
                "matched_patterns": missing_contracts,
                "preferred": (
                    f"sample_size <= {PAIRWISE_EVALUATION_MAX_SAMPLE_SIZE}",
                    "random_state=<declared deterministic seed>",
                    "record sample size and seed",
                ),
                "rationale": (
                    f"The execution cohort has {n_stays} stays; full-pairwise "
                    "silhouette is quadratic and can exhaust the step wall clock. "
                    "Keep model fitting and label assignment on all rows, but "
                    "evaluate silhouette on a deterministic bounded sample."
                ),
            }
        )
    return violations


def _variable_referenced_in_code(var_name: str, code_lower: str) -> bool:
    """Return True iff ``var_name`` appears in ``code_lower`` as an identifier.

    Uses word-boundary regex to avoid false positives — e.g. ``gcs`` should
    not match ``gcsscaler`` or ``foo_gcs_helper`` should match
    (boundary-separated). We allow a leading/trailing alphanumeric-underscore
    boundary so ``df["gcs"]`` and ``gcs_max`` both register the variable.
    """
    return (
        re.search(
            rf"(?<![A-Za-z0-9_]){re.escape(var_name.lower())}(?![A-Za-z0-9_])",
            code_lower,
        )
        is not None
    )


def _callable_path(
    node: ast.AST,
    estimator_aliases: Dict[str, str],
) -> str:
    """Return a normalized callable path, resolving assigned estimators."""

    if isinstance(node, ast.Name):
        return estimator_aliases.get(node.id, node.id).casefold()
    if isinstance(node, ast.Attribute):
        parent = _callable_path(node.value, estimator_aliases)
        return f"{parent}.{node.attr.casefold()}" if parent else node.attr.casefold()
    if isinstance(node, ast.Call):
        return _callable_path(node.func, estimator_aliases)
    return ""


def _normalized_pattern(pattern: object) -> str:
    return "".join(
        character for character in str(pattern).casefold() if character.isalnum()
    )


def _call_matches_forbidden_pattern(
    call: ast.Call,
    pattern: object,
    estimator_aliases: Dict[str, str],
) -> bool:
    raw_path = _callable_path(call.func, estimator_aliases)
    path = _normalized_pattern(raw_path)
    token = _normalized_pattern(pattern)
    if not path or not token:
        return False
    # These two are deliberately prose, not callables: they describe a reporting
    # habit and a misuse, so they stay substring tests over the whole path.
    if token.startswith("reportmean"):
        return "mean" in path
    if token == "kmeansonbinary":
        return "kmeans" in path
    # Every other pattern names a CALLABLE, so it must match a name, not a
    # substring of one. The naked `token in path` test fired exactly once in the
    # whole recorded corpus and that once was wrong: `ols(` matched
    # `matched_controls.append(...)` -- "contr-OLS" -- in a propensity-score
    # script, told the agent it had run ordinary least squares on a binary
    # outcome, and burned both repairs on a call that did not exist. The step
    # died with zero true positives ever found by this rule.
    #
    # Comparing dotted segments keeps every real hit -- `sm.ols`, `np.mean`,
    # `x.mean()`, `sklearn.cluster.KMeans`, `LinearRegression()` -- and drops
    # the accidental ones, including the latent twin of the same bug where
    # `.mean()` matched a helper called `weighted_mean`.
    segments = {_normalized_pattern(part) for part in raw_path.split(".")}
    segments.discard("")
    return token in segments or token == path


def _node_variable_sources(
    node: ast.AST,
    *,
    variable_names: Dict[str, str],
    assignment_sources: Dict[str, set[str]],
) -> set[str]:
    """Resolve cohort variables actually flowing into one AST expression."""

    sources: set[str] = set()
    for item in ast.walk(node):
        if isinstance(item, ast.Name):
            folded = item.id.casefold()
            if folded in variable_names:
                sources.add(folded)
            sources.update(assignment_sources.get(item.id, ()))
        elif isinstance(item, ast.Constant) and isinstance(item.value, str):
            text = item.value.casefold()
            for folded in variable_names:
                if re.search(
                    rf"(?<![A-Za-z0-9_]){re.escape(folded)}(?![A-Za-z0-9_])",
                    text,
                ):
                    sources.add(folded)
    return sources


def _ast_forbidden_pattern_hits(
    code: str,
    *,
    variables: Sequence[ConceptDescriptor],
) -> Dict[str, set[str]]:
    """Bind forbidden methods to variables that reach the method call.

    Comments, imports, docstrings, and unrelated calls cannot create a hit.
    Syntax-invalid scripts are left to the syntax gate rather than converted
    into a scientifically misleading method-compatibility error.
    """

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {}

    variable_names = {variable.name.casefold(): variable.name for variable in variables}
    assignment_sources: Dict[str, set[str]] = {}
    estimator_aliases: Dict[str, str] = {}
    assignments: list[tuple[list[str], ast.AST]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.NamedExpr)):
            if isinstance(node, ast.Assign):
                targets = node.targets
                value = node.value
            else:
                targets = [node.target]
                value = node.value
            names = [target.id for target in targets if isinstance(target, ast.Name)]
            if not names or value is None:
                continue
            assignments.append((names, value))
            if isinstance(value, ast.Call):
                constructor = _callable_path(value.func, estimator_aliases)
                for name in names:
                    estimator_aliases[name] = constructor

    for _ in range(len(assignments) + 1):
        changed = False
        for names, value in assignments:
            sources = _node_variable_sources(
                value,
                variable_names=variable_names,
                assignment_sources=assignment_sources,
            )
            for name in names:
                prior = assignment_sources.setdefault(name, set())
                if not sources.issubset(prior):
                    prior.update(sources)
                    changed = True
        if not changed:
            break

    hits: Dict[str, set[str]] = {}
    variable_by_name = {variable.name.casefold(): variable for variable in variables}
    for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
        sources = _node_variable_sources(
            call,
            variable_names=variable_names,
            assignment_sources=assignment_sources,
        )
        for source in sources:
            variable = variable_by_name[source]
            kind = _variable_kind(variable)
            rule = FORBIDDEN_METHOD_BY_KIND.get(kind or "")
            if not rule:
                continue
            matched = {
                str(pattern)
                for pattern in rule["forbidden_patterns"]  # type: ignore[union-attr]
                if _call_matches_forbidden_pattern(
                    call,
                    pattern,
                    estimator_aliases,
                )
            }
            if matched:
                hits.setdefault(source, set()).update(matched)
    return hits


def _ast_call_pattern_hits(
    code: str,
    patterns: Sequence[object],
) -> set[str]:
    """Return forbidden callable patterns present in executable AST nodes."""

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set()
    estimator_aliases: Dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign, ast.NamedExpr)):
            continue
        if isinstance(node, ast.Assign):
            targets = node.targets
            value = node.value
        else:
            targets = [node.target]
            value = node.value
        if not isinstance(value, ast.Call):
            continue
        constructor = _callable_path(value.func, estimator_aliases)
        for target in targets:
            if isinstance(target, ast.Name):
                estimator_aliases[target.id] = constructor
    return {
        str(pattern)
        for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call))
        for pattern in patterns
        if _call_matches_forbidden_pattern(call, pattern, estimator_aliases)
    }


def detect_forbidden_pattern_usage(
    code: str,
    context: ResearchContext,
    step: Optional[AnalysisStep] = None,
) -> List[Dict[str, object]]:
    """Scan generated code for forbidden method patterns over constrained variables.

    Returns a list of violation dicts (empty when the code is clean).

    Each violation records the offending variable, its kind, the
    matched forbidden pattern (lowercased) and the preferred
    alternatives + rationale from the matrix. Callers pass the
    violation list to :func:`format_violation_message` to produce a
    coder-repair prompt; callers can also use the dict-level fields
    to log structured audit entries.

    Method-family checks are bound to executable AST calls and the cohort
    variables flowing into those calls. Comments, docstrings, imports, and
    unrelated expressions cannot grant a blocking finding. Syntax-invalid code
    remains the syntax gate's responsibility.

    The function is intentionally **non-blocking** — it returns
    findings, never mutates the code or raises. Pipeline glue is
    responsible for deciding whether to repair, fail-closed or
    proceed.
    """
    if not code:
        return []
    code_lower = code.lower()
    violations: List[Dict[str, object]] = _helper_call_contract_violations(code)
    violations.extend(
        _large_cohort_silhouette_violations(
            code,
            n_stays=int(context.cohort.n_stays),
        )
    )
    structural_hits = _ast_forbidden_pattern_hits(
        code,
        variables=context.variables,
    )
    declared_outcomes = {
        str(name or "").strip().casefold()
        for name in (
            context.target_outcome,
            *(context.cohort.outcome_columns or ()),
        )
        if str(name or "").strip()
    }
    selected_trajectory = {
        variable.name
        for variable in selected_trajectory_variables(
            context=context,
            script_text=code,
            step=step,
        )
    }
    for var in context.variables:
        kind = _variable_kind(var)
        if not kind:
            continue
        if kind in {"binary", "count"} and not (
            var.role == VariableRole.OUTCOME
            or var.name.strip().casefold() in declared_outcomes
        ):
            continue
        if (
            var.name not in selected_trajectory
            and var.name.casefold() not in structural_hits
        ):
            continue
        rule = FORBIDDEN_METHOD_BY_KIND.get(kind)
        if not rule:
            continue
        matched = sorted(structural_hits.get(var.name.casefold(), ()))
        if not matched and var.name in selected_trajectory:
            matched = sorted(
                _ast_call_pattern_hits(
                    code,
                    rule["forbidden_patterns"],  # type: ignore[arg-type]
                )
            )
        if matched:
            violations.append(
                {
                    "variable": var.name,
                    "kind": kind,
                    "matched_patterns": matched,
                    "preferred": rule["preferred"],
                    "rationale": rule["rationale"],
                }
            )
    if selected_trajectory and trajectory_zero_imputation_detected(
        code,
        trajectory_columns=selected_trajectory,
    ):
        violations.append(
            {
                "variable": ", ".join(sorted(selected_trajectory)),
                "kind": "fixed_window_trajectory",
                "matched_patterns": ["zero_imputation"],
                "preferred": (
                    "agent_declared_non_zero_missingness_representation",
                    "observed_window_membership_rule",
                ),
                "rationale": (
                    "An unobserved or trailing trajectory window is not an "
                    "observed zero state."
                ),
            }
        )
    if step is not None:
        roles = trajectory_step_roles(step)
        downstream_roles = roles & {
            "candidate_selection",
            "stability_freeze",
        }
        if "representation" in roles and not downstream_roles:
            role_patterns = [
                pattern
                for pattern in (
                    "kmeans",
                    "minibatchkmeans",
                    "gaussianmixture",
                    "agglomerativeclustering",
                    "dbscan",
                    "hdbscan",
                    "fit_predict",
                    "cluster_selection",
                    "cluster_stability",
                    "cluster_assignments",
                    "cluster_profile",
                    "cluster_outcome",
                    "outcome_by_cluster",
                )
                if re.search(
                    rf"(?<![a-z0-9]){re.escape(pattern)}(?![a-z0-9])",
                    code_lower,
                )
            ]
            if role_patterns:
                violations.append(
                    {
                        "variable": step.step_id,
                        "kind": "trajectory_role_scope",
                        "matched_patterns": sorted(set(role_patterns)),
                        "preferred": (
                            "representation_artifact_only",
                            "trajectory_membership",
                        ),
                        "rationale": (
                            "The representation owner may transform and audit "
                            "features, but cluster fitting/selection/stability and "
                            "characterization belong to downstream agent-planned roles."
                        ),
                    }
                )
    return violations


def format_violation_message(violations: List[Dict[str, object]]) -> str:
    """Render violations into a structured error message for ``CoderAgent.repair``.

    The message is intentionally written in the *same shape* as a
    runtime traceback header so the existing ``code_repair`` pathway
    treats it uniformly: a problem statement, an itemised cause, and
    an explicit instruction for the corrective rewrite.
    """
    if not violations:
        return ""
    lines = [
        "PRE-EXECUTION COMPATIBILITY CHECK FAILED.",
        "",
        "Your script uses one or more incompatible analytical methods or "
        "violates a documented method-helper call contract. The deterministic "
        "check is scoped to variables and helper calls actually referenced in "
        "the script; see the method-compatibility and helper API contracts.",
        "",
        "Violations:",
    ]
    for v in violations:
        patterns = ", ".join(str(p) for p in v["matched_patterns"])  # type: ignore[index]
        preferred = ", ".join(str(p) for p in v["preferred"])  # type: ignore[index]
        lines.append(
            f"  - Variable `{v['variable']}` (kind: {v['kind']}): "
            f"matched forbidden pattern(s) `{patterns}`. "
            f"Use one of: {preferred}. "
            f"Reason: {v['rationale']}"
        )
    lines.append("")
    lines.append(
        "Rewrite the script using kind-appropriate methods and exact documented "
        "helper signatures. Do not pick a different variant of the same forbidden "
        "family (e.g. switching from `KMeans` to `MiniBatchKMeans` does not "
        "satisfy the check)."
    )
    return "\n".join(lines)


__all__ = [
    "FORBIDDEN_METHOD_BY_KIND",
    "PAIRWISE_EVALUATION_FULL_COHORT_MAX_ROWS",
    "PAIRWISE_EVALUATION_MAX_SAMPLE_SIZE",
    "detect_forbidden_pattern_usage",
    "format_violation_message",
    "render_computational_budget_constraints",
    "render_variable_constraints",
    "variable_kind_constraints",
]
