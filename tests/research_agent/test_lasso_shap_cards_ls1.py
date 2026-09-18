"""First Tool Cards: deterministic lasso selection + SHAP attribution.

Kernel modules under test: ``easyicu.research_agent.methods.lasso_selection``
and ``easyicu.research_agent.methods.shap_attribution``.

Reproduction-honesty note (read before reusing these cards): the
``independent_reproduction`` evidence below is a *deterministic rerun on
fixed data under a different process-global RNG state with the kernel's own
seed configuration pinned*.  It verifies byte-reproducibility; it is NOT a
fresh data draw and, for the lasso card, NOT a different CV partition.  In
particular, ``LassoCV`` legitimately changes its chosen alpha (and therefore
its output bytes) when its KFold ``random_state`` changes -- cross-seed byte
stability is not claimed for the ``"lassocv"`` path, and the lasso card's
origin run accordingly uses the fixed-alpha ``"lasso"`` path, whose
``cyclic`` coordinate descent is seed-invariant by construction.  A dedicated
test below pins that numeric seed-invariance claim.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from easyicu.research_agent.authority.tool_promotion import (
    ApplicabilityAttestation,
    ReproductionAttempt,
    decide_tool_promotion,
    require_verified_tool,
)
from easyicu.research_agent.canonical_json import canonical_sha256
from easyicu.research_agent.methods import lasso_selection as lasso_mod
from easyicu.research_agent.methods import shap_attribution as shap_mod
from easyicu.research_agent.methods.lasso_selection import (
    LassoSelectionError,
    lasso_select,
    result_sha256 as lasso_sha256,
)
from easyicu.research_agent.methods.shap_attribution import (
    ShapAttributionError,
    result_sha256 as shap_sha256,
    shap_attribute,
)
from easyicu.research_agent.methods.tool_card import (
    ToolCard,
    tool_card_completeness_issues,
    tool_card_sha256,
)
from easyicu.research_agent.planning import capability_registry as registry_module

LASSO_EXECUTOR_MODULE = "easyicu.research_agent.methods.lasso_selection"
SHAP_EXECUTOR_MODULE = "easyicu.research_agent.methods.shap_attribution"

TRUE_VARS = ("x0", "x1", "x3")


def _clear_grants() -> None:
    registry_module._TOOL_CARD_GRANTS.pop(LASSO_EXECUTOR_MODULE, None)
    registry_module._TOOL_CARD_GRANTS.pop(SHAP_EXECUTOR_MODULE, None)


def _orthogonal_fixture() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Orthogonal-Gaussian design with known support {x0, x1, x3}."""

    rng = np.random.RandomState(0)
    n, p = 200, 10
    base = rng.randn(n, p)
    ortho, _ = np.linalg.qr(base)
    design = ortho * np.sqrt(n)
    beta = np.zeros(p)
    beta[0] = 3.0
    beta[1] = -2.0
    beta[3] = 1.5
    outcome = design @ beta + 0.1 * rng.randn(n)
    return design, outcome, [f"x{i}" for i in range(p)]


def _xgb_fixture() -> tuple[np.ndarray, np.ndarray, list[str], object]:
    import xgboost as xgb

    rng = np.random.RandomState(0)
    features = rng.randn(60, 4)
    target = 2.0 * features[:, 0] - features[:, 1] + 0.1 * rng.randn(60)
    model = xgb.XGBRegressor(
        n_estimators=10,
        max_depth=3,
        learning_rate=0.3,
        random_state=0,
        n_jobs=1,
        subsample=1.0,
        colsample_bytree=1.0,
        reg_lambda=1.0,
    )
    model.fit(features, target)
    return features, target, ["a", "b", "c", "d"], model


# ---------------------------------------------------------------------------
# Lasso kernel: determinism, correctness, fail-closed
# ---------------------------------------------------------------------------


def test_lasso_fixed_alpha_digest_stable_across_reruns() -> None:
    design, outcome, names = _orthogonal_fixture()
    first = lasso_select(design, outcome, feature_names=names, random_state=0)
    second = lasso_select(design, outcome, feature_names=names, random_state=0)
    assert first.to_json() == second.to_json()
    assert lasso_sha256(first) == lasso_sha256(second)
    assert lasso_sha256(first) == canonical_sha256(first.to_json())


def test_lassocv_digest_stable_for_fixed_seed() -> None:
    design, outcome, names = _orthogonal_fixture()
    first = lasso_select(
        design, outcome, feature_names=names, method="lassocv", random_state=0
    )
    second = lasso_select(
        design, outcome, feature_names=names, method="lassocv", random_state=0
    )
    assert lasso_sha256(first) == lasso_sha256(second)
    assert set(TRUE_VARS) <= set(first.selected_vars)
    assert len(first.alphas) > 1
    assert len(first.cv_mean_mse) == len(first.alphas) == len(first.cv_std_mse)


def test_lasso_selects_true_vars_on_orthogonal_design() -> None:
    design, outcome, names = _orthogonal_fixture()
    result = lasso_select(
        design, outcome, feature_names=names, method="lasso", alpha=0.1
    )
    assert tuple(result.selected_vars) == TRUE_VARS
    assert result.alpha == pytest.approx(0.1)
    assert len(result.cv_mean_mse) == 1


def test_lasso_fixed_alpha_numerics_seed_invariant() -> None:
    """Different kernel seeds give identical fits on the fixed-alpha path.

    The fitted coefficients and selection are seed-invariant under
    ``cyclic`` updates; only the split-dependent CV MSE summary follows
    the pinned KFold seed, so it is compared for same-seed reruns only
    (see the digest test above).
    """

    design, outcome, names = _orthogonal_fixture()
    low = lasso_select(design, outcome, feature_names=names, random_state=0)
    high = lasso_select(design, outcome, feature_names=names, random_state=7)
    assert low.coefs == high.coefs
    assert low.selected_vars == high.selected_vars
    assert low.alpha == high.alpha


def test_lasso_fail_closed() -> None:
    design, outcome, names = _orthogonal_fixture()
    bad_frame = pd.DataFrame({"x0": ["a", "b", "c"], "x1": [1.0, 2.0, 3.0]})
    with pytest.raises(LassoSelectionError):
        lasso_select(bad_frame, [1.0, 2.0, 3.0])
    nan_y = np.array(outcome, dtype=float)
    nan_y[0] = np.nan
    with pytest.raises(LassoSelectionError):
        lasso_select(design, nan_y, feature_names=names)
    with pytest.raises(LassoSelectionError):
        lasso_select(np.empty((0, 3)), np.empty((0,)))
    with pytest.raises(LassoSelectionError):
        lasso_select(design, outcome[:-1], feature_names=names)
    with pytest.raises(LassoSelectionError):
        lasso_select(design, outcome, feature_names=names, alpha=0.0)
    with pytest.raises(LassoSelectionError):
        lasso_select(design, outcome, feature_names=names, method="elasticnet")
    with pytest.raises(LassoSelectionError):
        lasso_select(
            design, outcome, feature_names=names, method="lassocv", alphas=[]
        )
    with pytest.raises(LassoSelectionError):
        lasso_select(design[:4], outcome[:4], feature_names=names, cv=5)


# ---------------------------------------------------------------------------
# SHAP kernel: determinism, consistency, fail-closed
# ---------------------------------------------------------------------------


def test_shap_digest_stable_across_reruns() -> None:
    features, _, names, model = _xgb_fixture()
    first = shap_attribute(model, features, feature_names=names)
    second = shap_attribute(model, features, feature_names=names)
    assert first.to_json() == second.to_json()
    assert shap_sha256(first) == shap_sha256(second)


def test_shap_values_reconstruct_predictions() -> None:
    features, _, names, model = _xgb_fixture()
    result = shap_attribute(model, features, feature_names=names)
    matrix = np.asarray(result.shap_values)
    predicted = np.asarray(result.predictions)
    assert matrix.shape == (60, 4)
    np.testing.assert_allclose(matrix.sum(axis=1) + result.base_value, predicted,
                               rtol=0, atol=1e-4)
    assert len(result.mean_abs_shap) == 4
    assert all(value >= 0 for value in result.mean_abs_shap)
    assert int(np.argmax(result.mean_abs_shap)) == 0


def test_shap_rejects_non_tree_model_with_fallback_pointer() -> None:
    from sklearn.linear_model import LinearRegression

    features, target, names, _ = _xgb_fixture()
    model = LinearRegression().fit(features, target)
    with pytest.raises(ShapAttributionError, match="permutation_importance"):
        shap_attribute(model, features, feature_names=names)


def test_shap_fail_closed_on_bad_inputs() -> None:
    _, _, names, model = _xgb_fixture()
    _, _, _, ref_model = _xgb_fixture()
    with pytest.raises(ShapAttributionError):
        shap_attribute(ref_model, np.empty((0, 4)), feature_names=names)
    bad = np.ones((8, 4))
    bad[0, 0] = np.inf
    with pytest.raises(ShapAttributionError):
        shap_attribute(model, bad, feature_names=names)


# ---------------------------------------------------------------------------
# Tool Card 1: lasso variable selection
# ---------------------------------------------------------------------------


def _lasso_card(origin_digest: str) -> ToolCard:
    return ToolCard.model_validate(
        {
            "schema_version": "easyicu.tool_card/1",
            "tool_name": "lasso_variable_selection",
            "tool_version": lasso_mod.TOOL_VERSION,
            "method_meaning": (
                "L1-penalized linear variable selection at a pinned alpha "
                "with fixed KFold splits; |coef| above threshold selects"
            ),
            "inputs": [
                {"name": "X", "value_kind": "float_matrix",
                 "description": "complete-case numeric design matrix"},
                {"name": "y", "value_kind": "float_vector",
                 "description": "numeric outcome vector aligned to X rows"},
                {"name": "alpha", "value_kind": "float",
                 "required": False,
                 "description": "pinned L1 penalty for method=lasso"},
            ],
            "outputs": [
                {"name": "selected_vars", "value_kind": "string_list"},
                {"name": "coefs", "value_kind": "float_list"},
                {"name": "alpha", "value_kind": "float"},
                {"name": "cv_mean_mse", "value_kind": "float_list"},
            ],
            "population_assumption": (
                "complete-case numeric design matrices; origin run on a "
                "synthetic orthogonal-Gaussian fixture (seed 0, n=200, p=10)"
            ),
            "timing_assumption": (
                "retrospective analysis on a frozen data extract; no "
                "observation-window or real-time constraint"
            ),
            "origin_output_sha256": origin_digest,
            "validation_evidence": [
                {
                    "evidence_id": "lasso-sandbox-run-1",
                    "kind": "sandbox_run",
                    "passed": True,
                    "output_sha256": origin_digest,
                    "note": (
                        "origin run: fixed-alpha lasso, alpha=0.1, cv=5, "
                        "random_state=0 on the orthogonal fixture"
                    ),
                },
                {
                    "evidence_id": "lasso-repro-1",
                    "kind": "independent_reproduction",
                    "passed": True,
                    "output_sha256": origin_digest,
                    "note": (
                        "deterministic rerun on identical fixed data with "
                        "the kernel seed configuration pinned "
                        "(random_state=0) and a different process-global "
                        "RNG state (numpy seed 999 vs 0); output digest "
                        "byte-identical. NOT a fresh data draw and NOT a "
                        "different CV partition; cross-seed LassoCV "
                        "stability is explicitly not claimed."
                    ),
                },
                {
                    "evidence_id": "lasso-applicability-1",
                    "kind": "applicability_check",
                    "passed": True,
                    "output_sha256": origin_digest,
                    "note": (
                        "intended reuse is variable selection on "
                        "complete-case numeric matrices, inside the card "
                        "population; no clinical transport claimed"
                    ),
                },
            ],
        },
        strict=True,
    )


def test_lasso_tool_card_promotion_and_grant() -> None:
    _clear_grants()
    try:
        design, outcome, names = _orthogonal_fixture()
        origin = lasso_select(
            design, outcome, feature_names=names, method="lasso",
            alpha=0.1, cv=5, random_state=0,
        )
        origin_digest = lasso_sha256(origin)
        assert tuple(origin.selected_vars) == TRUE_VARS

        np.random.seed(999)
        try:
            rerun = lasso_select(
                design, outcome, feature_names=names, method="lasso",
                alpha=0.1, cv=5, random_state=0,
            )
        finally:
            np.random.seed()
        assert lasso_sha256(rerun) == origin_digest

        card = _lasso_card(origin_digest)
        assert tool_card_completeness_issues(card) == []

        decision = decide_tool_promotion(
            card=card,
            reproductions=[
                ReproductionAttempt.model_validate(
                    {
                        "attempt_id": "lasso-repro-1",
                        "output_sha256": lasso_sha256(rerun),
                        "passed": True,
                        "independent": True,
                    },
                    strict=True,
                )
            ],
            executor_module=LASSO_EXECUTOR_MODULE,
            applicability=ApplicabilityAttestation.model_validate(
                {
                    "intended_population": (
                        "complete-case numeric design matrices for "
                        "variable selection"
                    ),
                    "intended_timing": (
                        "retrospective analysis on a frozen extract"
                    ),
                    "covered": True,
                    "reason": (
                        "intended reuse matches the card population and "
                        "timing exactly; synthetic origin bounds the "
                        "claim to method mechanics"
                    ),
                },
                strict=True,
            ),
        )
        assert decision.granted_verified_tool is True
        assert decision.allowed_identity == "verified_tool"
        assert decision.card_sha256 == tool_card_sha256(card)
        require_verified_tool(decision)

        record = registry_module.register_tool_card_grant(
            executor_module=LASSO_EXECUTOR_MODULE,
            card=card,
            decision=decision,
        )
        assert record["tool_name"] == "lasso_variable_selection"
        assert record["tool_version"] == lasso_mod.TOOL_VERSION
        assert record["card_sha256"] == tool_card_sha256(card)
        assert (
            registry_module.granted_tool_identity(LASSO_EXECUTOR_MODULE)
            == "verified_tool"
        )
    finally:
        _clear_grants()


# ---------------------------------------------------------------------------
# Tool Card 2: SHAP attribution
# ---------------------------------------------------------------------------


def _shap_card(origin_digest: str) -> ToolCard:
    return ToolCard.model_validate(
        {
            "schema_version": "easyicu.tool_card/1",
            "tool_name": "shap_tree_attribution",
            "tool_version": shap_mod.TOOL_VERSION,
            "method_meaning": (
                "exact TreeExplainer SHAP values for a fixed fitted tree "
                "model; reports per-feature mean|SHAP| plus the SHAP "
                "matrix, base value and predictions for beeswarm/waterfall"
            ),
            "inputs": [
                {"name": "model", "value_kind": "fitted_tree_model",
                 "description": "fitted tree model with predict"},
                {"name": "X", "value_kind": "float_matrix",
                 "description": "complete-case numeric matrix"},
            ],
            "outputs": [
                {"name": "mean_abs_shap", "value_kind": "float_list"},
                {"name": "shap_values", "value_kind": "float_matrix"},
                {"name": "base_value", "value_kind": "float"},
                {"name": "predictions", "value_kind": "float_list"},
            ],
            "population_assumption": (
                "any complete-case numeric matrix scored by the fixed "
                "model; origin run on a synthetic fixture (seed 0, "
                "n=60, p=4, XGBRegressor 10 trees)"
            ),
            "timing_assumption": (
                "retrospective analysis on a frozen data extract; no "
                "observation-window or real-time constraint"
            ),
            "origin_output_sha256": origin_digest,
            "validation_evidence": [
                {
                    "evidence_id": "shap-sandbox-run-1",
                    "kind": "sandbox_run",
                    "passed": True,
                    "output_sha256": origin_digest,
                    "note": (
                        "origin run: exact TreeExplainer on the fixed "
                        "XGBRegressor and fixed 60x4 matrix"
                    ),
                },
                {
                    "evidence_id": "shap-repro-1",
                    "kind": "independent_reproduction",
                    "passed": True,
                    "output_sha256": origin_digest,
                    "note": (
                        "refit the XGBRegressor with identical pinned "
                        "hyperparameters (including random_state=0, "
                        "n_jobs=1) under a different process-global RNG "
                        "state (numpy seed 1234 vs 0), then reran exact "
                        "TreeExplainer on the fixed matrix; SHAP exact "
                        "values are a pure function of (model bytes, "
                        "data); output digest byte-identical"
                    ),
                },
                {
                    "evidence_id": "shap-applicability-1",
                    "kind": "applicability_check",
                    "passed": True,
                    "output_sha256": origin_digest,
                    "note": (
                        "intended reuse is attributing fixed tree-model "
                        "predictions on complete-case matrices, inside "
                        "the card population; no clinical transport claimed"
                    ),
                },
            ],
        },
        strict=True,
    )


def test_shap_tool_card_promotion_and_grant() -> None:
    _clear_grants()
    try:
        import xgboost as xgb

        features, target, names, model = _xgb_fixture()
        origin = shap_attribute(model, features, feature_names=names)
        origin_digest = shap_sha256(origin)

        np.random.seed(1234)
        try:
            refit = xgb.XGBRegressor(
                n_estimators=10,
                max_depth=3,
                learning_rate=0.3,
                random_state=0,
                n_jobs=1,
                subsample=1.0,
                colsample_bytree=1.0,
                reg_lambda=1.0,
            )
            refit.fit(features, target)
            rerun = shap_attribute(refit, features, feature_names=names)
        finally:
            np.random.seed()
        assert shap_sha256(rerun) == origin_digest

        card = _shap_card(origin_digest)
        assert tool_card_completeness_issues(card) == []

        decision = decide_tool_promotion(
            card=card,
            reproductions=[
                ReproductionAttempt.model_validate(
                    {
                        "attempt_id": "shap-repro-1",
                        "output_sha256": shap_sha256(rerun),
                        "passed": True,
                        "independent": True,
                    },
                    strict=True,
                )
            ],
            executor_module=SHAP_EXECUTOR_MODULE,
            applicability=ApplicabilityAttestation.model_validate(
                {
                    "intended_population": (
                        "complete-case numeric matrices scored by a "
                        "fixed tree model"
                    ),
                    "intended_timing": (
                        "retrospective analysis on a frozen extract"
                    ),
                    "covered": True,
                    "reason": (
                        "intended reuse matches the card population and "
                        "timing exactly; synthetic origin bounds the "
                        "claim to attribution mechanics"
                    ),
                },
                strict=True,
            ),
        )
        assert decision.granted_verified_tool is True
        assert decision.allowed_identity == "verified_tool"
        assert decision.card_sha256 == tool_card_sha256(card)
        require_verified_tool(decision)

        record = registry_module.register_tool_card_grant(
            executor_module=SHAP_EXECUTOR_MODULE,
            card=card,
            decision=decision,
        )
        assert record["tool_name"] == "shap_tree_attribution"
        assert record["tool_version"] == shap_mod.TOOL_VERSION
        assert record["card_sha256"] == tool_card_sha256(card)
        assert (
            registry_module.granted_tool_identity(SHAP_EXECUTOR_MODULE)
            == "verified_tool"
        )
    finally:
        _clear_grants()
