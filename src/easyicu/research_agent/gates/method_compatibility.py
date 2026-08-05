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
    return None


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
    imported_names = {"silhouette_score"}
    fixed_seed_names: set[str] = set()
    for node in ast.walk(tree):
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
            and isinstance(node.iter, (ast.List, ast.Set, ast.Tuple))
            and node.iter.elts
            and all(
                isinstance(item, ast.Constant)
                and isinstance(item.value, int)
                and not isinstance(item.value, bool)
                for item in node.iter.elts
            )
        ):
            fixed_seed_names.add(node.target.id)

    violations: List[Dict[str, object]] = []
    for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
        callable_path = _callable_path(call.func, {})
        terminal_name = callable_path.rsplit(".", 1)[-1]
        if terminal_name not in {name.casefold() for name in imported_names}:
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
        random_state = keywords.get("random_state")
        deterministic_random_state = (
            isinstance(random_state, ast.Name)
            and random_state.id in fixed_seed_names
        ) or (
            random_state is not None
            and _static_integer_upper_bound(
                random_state,
                assignments=assignments,
            )
            is not None
        )
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
