"""Tested in-tree statistical kernels the Coder may import instead of re-deriving.

Why this module exists
----------------------
Measured 2026-07-30: six modules under ``research_agent/methods/`` -- 1,430
lines, 49 passing tests -- had **zero importers anywhere outside their own test
files**.  Not in ``src/``, not in ``tools/``, not in ``benchmarks/``.  Five of
them were simultaneously declared ``implementation="planned"`` in
``planning/analysis_method_suite.py``, which is the surface the Planner reads.
The map said "not implemented", so nothing ever routed to them, so a Coder
asked for a DeLong interval re-derived the variance by hand.

The cause is structural, not clerical.  Every drift guard in
``test_analysis_method_suite.py`` iterates *the map* and checks reality matches
it (``for suite, m in _ALL_METHODS``).  That direction catches claiming MORE
than we have.  Nothing iterated the *code* and checked the map mentions it, so
claiming LESS than we have -- dead code -- was invisible by construction.

This module closes the other direction.  It is the declared inventory of
kernels that are reachable *through the Coder* rather than through a host
runner, and ``tests/research_agent/test_method_kernel_reachability.py`` asserts
that every module under ``methods/`` is reachable by exactly one of the two
routes:

* a non-test importer inside ``src/`` (the host calls it), or
* an entry in :data:`CURATED_METHOD_KERNELS` (the Coder may call it).

A module in neither is dead, and the test fails naming it.  That is the
long-lived map: not a document someone must remember to update, but a check
that fails when the code and the declaration disagree.

Why the Coder and not a host runner
-----------------------------------
Wiring one of these the way ``finalize.py`` wires ``compute_e_value`` costs
~110 lines of bespoke CSV plumbing per method, including column-name guessing
(``"odds_ratio", "or", "OR"``).  Five of those is 550 new lines of the exact
shape this project keeps paying for.  These kernels are already inside the
runner image and already byte-verified by it: ``DockerRunner`` hashes every
``.py`` under ``research_agent/`` and refuses to run if it does not match the
host tree.  So the import already works; only the *declaration* was missing.

This does not make the kernels deterministic.  The Coder still writes the call,
so the result is still value-verified like any Coder output.  What changes is
that it calls 335 reviewed, tested lines instead of re-deriving Schoenfeld
residuals inside a generated script.

Deliberately data-only, mirroring ``method_packages.py``: authority code binds
the declared set into a reproducibility fingerprint without importing execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

__all__ = [
    "MethodKernel",
    "CURATED_METHOD_KERNELS",
    "KERNEL_MODULE_NAMES",
    "UnreachableKernel",
    "DECLARED_UNREACHABLE_KERNELS",
    "UNREACHABLE_MODULE_NAMES",
]

_KERNEL_PACKAGE = "easyicu.research_agent.methods"


@dataclass(frozen=True)
class MethodKernel:
    """One reviewed in-tree kernel offered to Coder-generated analysis code."""

    module: str  # module name under research_agent.methods
    entrypoints: Tuple[str, ...]  # public callables; existence is asserted
    capability: str  # what it computes, in the Coder's vocabulary
    families: Tuple[str, ...]  # analysis families it applies to
    fallback: str  # what the Coder must do if it is unavailable
    requires: Tuple[str, ...] = ()  # packages the kernel itself imports

    @property
    def import_path(self) -> str:
        return f"{_KERNEL_PACKAGE}.{self.module}"


CURATED_METHOD_KERNELS: Tuple[MethodKernel, ...] = (
    MethodKernel(
        module="survival_inputs",
        requires=("numpy", "pandas"),
        entrypoints=("event_time_reconciliation_receipt", "SurvivalInputError"),
        capability=(
            "fail-closed reconciliation of a declared event code against its "
            "event time before a risk set is built — refuses an event row whose "
            "time cannot place it on the follow-up axis, and an event code "
            "outside the declared closed set, returning counts only so it "
            "cannot change the analysis population. A censored row with no "
            "event time is the expected shape and is reported, not refused"
        ),
        families=("time_to_event", "survival", "causal_inference"),
        fallback=(
            "reconcile the pair in the script and report both counts in the "
            "step's own output: an event with no usable time recoded to 'no "
            "event' moves it into the survivor arm, and dropped through "
            "duration/event missingness it leaves both arms"
        ),
    ),
    MethodKernel(
        module="ph_schoenfeld",
        requires=("pandas", "lifelines"),
        entrypoints=("ph_test", "run_ph_test", "PHTestResult"),
        capability=(
            "proportional-hazards check for a fitted Cox model — Schoenfeld "
            "residual test per covariate and globally, returning chi2 and p"
        ),
        families=("time_to_event", "survival"),
        fallback=(
            "lifelines.statistics.proportional_hazard_test directly (this "
            "kernel wraps it and normalises the result shape)"
        ),
    ),
    MethodKernel(
        module="delong_auc",
        requires=("numpy", "scipy"),
        entrypoints=(
            "delong_auc_ci",
            "delong_auc_variance",
            "delong_test",
            "AUCResult",
        ),
        capability=(
            "DeLong confidence interval on an AUROC, and the DeLong test "
            "comparing two correlated AUROCs on the same cases"
        ),
        families=("prediction", "prediction_model", "dynamic_prediction"),
        fallback=(
            "bootstrap percentile CI on the AUROC — scikit-learn has no DeLong "
            "implementation, so hand-deriving the variance is the alternative"
        ),
    ),
    MethodKernel(
        module="rmst",
        requires=("numpy", "scipy"),
        entrypoints=("rmst", "rmst_difference", "RMSTResult"),
        capability=(
            "restricted mean survival time up to a horizon, and the between-"
            "group RMST difference with its sampling standard error"
        ),
        families=("time_to_event", "survival"),
        fallback=(
            "NOT lifelines.utils.restricted_mean_survival_time(return_variance"
            "=True) — that returns the population variance of the restricted "
            "survival time, not the estimator's sampling SE, and inflates the "
            "CI by ~sqrt(n). This kernel computes the integral-form variance."
        ),
    ),
    MethodKernel(
        module="decision_curve",
        requires=("numpy", "pandas"),
        entrypoints=(
            "net_benefit_curve",
            "net_benefit_at",
            "summarize_decision_curve",
            "DecisionCurveResult",
        ),
        capability=(
            "decision-curve analysis — net benefit across threshold "
            "probabilities against treat-all and treat-none"
        ),
        families=("prediction", "prediction_model", "dynamic_prediction"),
        fallback="hand-computed net benefit at each threshold",
    ),
    MethodKernel(
        module="temporal_features",
        requires=("numpy", "pandas"),
        entrypoints=("onset_times", "incident_outcome_cohort", "landmark_cohort"),
        capability=(
            "timing primitives over the long trajectory (TRAJECTORY_PARQUET): "
            "first time a concept crosses a threshold; prevalent / incident / "
            "event-free classification relative to an index event; and the "
            "at-risk set plus follow-up clock from a landmark time"
        ),
        # Deliberately NOT "association". Measured 2026-07-30: with it declared,
        # this kernel out-ranked statsmodels as the top software resource for a
        # plain "fit an adjusted logistic regression" step, where trajectory
        # timing is not the question. A family list is a relevance claim, and
        # claiming a family too widely crowds out the tool the step needs.
        families=(
            "time_to_event",
            "survival",
            "causal_emulation",
            "dynamic_prediction",
        ),
        fallback=(
            "re-deriving onsets from the trajectory inside the analysis script "
            "— the failure this module was written for, where a 'measured but "
            "event-absent' row (e.g. a stage-0 record) was counted as an onset"
        ),
    ),
    MethodKernel(
        module="dynamic_prediction",
        requires=("numpy", "pandas", "sklearn"),
        entrypoints=(
            "build_landmark_feature_matrix",
            "attach_landmark_outcomes",
            "evaluate_landmark_probabilities",
            "DynamicPredictionEvaluation",
        ),
        capability=(
            "leakage-safe landmark feature slices, observable future-horizon "
            "labels, and per-landmark AUROC/Brier/calibration evaluation"
        ),
        families=("dynamic_prediction",),
        fallback=(
            "no ad-hoc fallback: hand-written row slicing can leak post-landmark "
            "measurements or label censored horizons as non-events"
        ),
    ),
    MethodKernel(
        module="conformal",
        requires=("numpy",),
        entrypoints=(
            "conformal_calibrate",
            "conformal_predict_sets",
            "conformal_evaluate",
            "ConformalResult",
        ),
        capability=(
            "split-conformal prediction sets with marginal (and Mondrian "
            "class-conditional) coverage guarantees"
        ),
        families=("prediction", "prediction_model"),
        fallback="no distribution-free coverage guarantee",
    ),
)


KERNEL_MODULE_NAMES: frozenset = frozenset(k.module for k in CURATED_METHOD_KERNELS)


@dataclass(frozen=True)
class UnreachableKernel:
    """A kernel module deliberately reachable by neither route, and why.

    This is NOT an escape hatch for "wire it later".  It exists because the
    reachability guard has exactly two honest answers -- the host calls it, or
    the Coder may call it -- and a third state is real: code whose resolution is
    a decision someone must take, not a wiring task.  Making that state explicit
    and reason-bearing is what keeps it from silently becoming the default.

    An entry costs a written reason and a named pending decision, so adding one
    is visible in review; leaving a module out of all three lists fails the
    guard.
    """

    module: str
    reason: str  # why it is not reachable today; asserted non-empty
    pending_decision: str  # what must be decided; asserted non-empty


# Empty, and that is the intended steady state rather than a gap.
#
# Its only entry was ``evalue``: a second E-value kernel that agreed with the
# wired ``sensitivity.compute_e_value`` on RR and HR but disagreed on OR -> RR,
# so the same odds ratio produced two different E-values depending on which
# module a caller reached. Its ``pending_decision`` was "which OR -> RR
# convention the reported E-value uses".
#
# Decided 2026-08-07: the observed-prevalence (Zhang-Yu) conversion in
# ``sensitivity.compute_e_value``, because it reads the cohort's own event rate
# instead of asking the caller to assert a rare- or common-outcome regime, and
# refuses rather than guessing when that rate is unavailable. ``evalue.py`` was
# deleted; the four properties only its tests covered moved to
# ``test_evalue_observed_baseline.py``.
DECLARED_UNREACHABLE_KERNELS: Tuple[UnreachableKernel, ...] = ()


UNREACHABLE_MODULE_NAMES: frozenset = frozenset(
    k.module for k in DECLARED_UNREACHABLE_KERNELS
)
