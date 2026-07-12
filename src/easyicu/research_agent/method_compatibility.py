"""Generic variable-kind × analytical-method compatibility matrix.

This module encodes statistical / clinical truths about *kinds* of
variables (ordinal, binary, count, right-skewed continuous) and the
method families that are inappropriate for each. It is consumed by
:func:`render_variable_constraints` to produce a self-review checklist
that the agent layer appends to every research-context summary, so the
planner / coder / analyzer / writer all see the same up-front rule
list **before** generated code reaches the post-hoc validators in
``audits/``.

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

from .schema import (
    AggregationRule,
    AnalysisStep,
    ConceptDescriptor,
    ResearchContext,
    VariableRole,
)
from .audits.patterns import (
    _DISTANCE_BASED_ESTIMATORS,
    _LINEAR_PCA_ESTIMATORS,
)
from .trajectory_contract import (
    is_continuous_trajectory_representation,
    selected_trajectory_variables,
    trajectory_zero_imputation_detected,
)


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
_RIGHT_SKEWED_LAB_NAMES = frozenset({
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
})


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
    if AggregationRule.SUM in (var.allowed_aggregations or []):
        return "count"
    if var.role == VariableRole.LAB and var.name.lower() in _RIGHT_SKEWED_LAB_NAMES:
        return "right_skewed_continuous"
    return None


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def variable_kind_constraints(
    variables: Sequence[ConceptDescriptor],
) -> List[Dict[str, object]]:
    """Return a list of per-variable constraint dicts (machine-readable)."""
    out: List[Dict[str, object]] = []
    for var in variables:
        kind = _variable_kind(var)
        if not kind:
            continue
        rule = FORBIDDEN_METHOD_BY_KIND.get(kind)
        if not rule:
            continue
        out.append({
            "variable": var.name,
            "kind": kind,
            "forbidden_patterns": rule["forbidden_patterns"],
            "preferred": rule["preferred"],
            "rationale": rule["rationale"],
        })
    return out


def render_variable_constraints(context: ResearchContext) -> str:
    """Return a human-readable self-review block for agent system prompts.

    Empty string when no variable in ``context.variables`` triggers a
    matrix entry; in that case the agent prompts are unchanged. This
    keeps the addition invisible for cohorts that don't carry any
    ordinal / count / binary / right-skewed-continuous variables (e.g.
    purely continuous-vital studies).
    """
    rules = variable_kind_constraints(context.variables)
    if not rules:
        return ""
    lines = [
        "Variable-type method-compatibility self-review checklist (derived",
        "automatically from the research context — must be honoured when",
        "drafting plans and writing code):",
    ]
    for r in rules:
        forbidden_head = ", ".join(list(r["forbidden_patterns"])[:5])  # type: ignore[index]
        preferred = ", ".join(list(r["preferred"]))  # type: ignore[index]
        lines.append(
            f"  - `{r['variable']}` (kind: {r['kind']}): DO NOT use "
            f"{forbidden_head}. PREFERRED: {preferred}. "
            f"Reason: {r['rationale']}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Post-codegen pre-execution enforcement (Patch C)
# ---------------------------------------------------------------------------


import re


def _variable_referenced_in_code(var_name: str, code_lower: str) -> bool:
    """Return True iff ``var_name`` appears in ``code_lower`` as an identifier.

    Uses word-boundary regex to avoid false positives — e.g. ``gcs`` should
    not match ``gcsscaler`` or ``foo_gcs_helper`` should match
    (boundary-separated). We allow a leading/trailing alphanumeric-underscore
    boundary so ``df["gcs"]`` and ``gcs_max`` both register the variable.
    """
    return re.search(rf"(?<![A-Za-z0-9_]){re.escape(var_name.lower())}(?![A-Za-z0-9_])", code_lower) is not None


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

    The check uses substring matching on the lowercased code. AST
    parsing would be cleaner but would refuse to scan code that
    fails to parse — and one of the main use cases is "the LLM
    produced something close to valid but not quite". Substring
    matching gives us a useful signal even when the code is slightly
    malformed.

    The function is intentionally **non-blocking** — it returns
    findings, never mutates the code or raises. Pipeline glue is
    responsible for deciding whether to repair, fail-closed or
    proceed.
    """
    if not code or not context.variables:
        return []
    code_lower = code.lower()
    violations: List[Dict[str, object]] = []
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
        if (
            var.name not in selected_trajectory
            and not _variable_referenced_in_code(var.name, code_lower)
        ):
            continue
        rule = FORBIDDEN_METHOD_BY_KIND.get(kind)
        if not rule:
            continue
        matched: List[str] = []
        for pat in rule["forbidden_patterns"]:  # type: ignore[union-attr]
            if pat.lower() in code_lower:
                matched.append(pat)
        if matched:
            violations.append({
                "variable": var.name,
                "kind": kind,
                "matched_patterns": matched,
                "preferred": rule["preferred"],
                "rationale": rule["rationale"],
            })
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
        "Your script uses one or more analytical methods that are forbidden "
        "for the variable kinds present in the research context. The check "
        "is a deterministic substring scan over the variables you actually "
        "referenced in the script; see Methods / Tier-assignment rule and "
        "the variable-type method-compatibility checklist for the policy.",
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
        "Rewrite the script using kind-appropriate methods. Do not pick a "
        "different variant of the same forbidden family (e.g. switching "
        "from `KMeans` to `MiniBatchKMeans` does not satisfy the check)."
    )
    return "\n".join(lines)


__all__ = [
    "FORBIDDEN_METHOD_BY_KIND",
    "detect_forbidden_pattern_usage",
    "format_violation_message",
    "render_variable_constraints",
    "variable_kind_constraints",
]
