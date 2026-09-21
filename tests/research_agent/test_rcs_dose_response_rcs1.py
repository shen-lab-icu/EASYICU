"""Restricted cubic spline dose-response kernel + its Tool Card.

Kernel module under test:
``easyicu.research_agent.methods.rcs_dose_response``.

Reproduction-honesty note (read before reusing this card): the
``independent_reproduction`` evidence below is a *deterministic rerun on
fixed data under a different process-global RNG state with the kernel's
own seed configuration pinned*.  It verifies exact local reruns and the
portable synthetic-origin receipt; it is
NOT a fresh data draw and NOT a different knot placement.  Knot count and
positions are modelling choices: any reuse must report them alongside the
numbers (the typed results always carry ``knots``), and the boundary
linearity outside the outer knots is an imposed constraint, not an
empirical finding.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from easyicu.research_agent.authority.tool_promotion import (
    ApplicabilityAttestation,
    ReproductionAttempt,
    decide_tool_promotion,
    require_verified_tool,
)
from easyicu.research_agent.canonical_json import canonical_sha256
from easyicu.research_agent.methods import rcs_dose_response as rcs_mod
from easyicu.research_agent.methods.rcs_dose_response import (
    RCSBasis,
    RCSError,
    RCSFitResult,
    nonlinearity_wald_test,
    predict_curve,
    rcs_basis,
    rcs_fit,
    result_sha256 as rcs_sha256,
)
from easyicu.research_agent.methods.tool_card import (
    ToolCard,
    synthetic_origin_sha256,
    tool_card_completeness_issues,
    tool_card_sha256,
)
from easyicu.research_agent.planning import capability_registry as registry_module

RCS_EXECUTOR_MODULE = "easyicu.research_agent.methods.rcs_dose_response"


def _clear_grant() -> None:
    registry_module._TOOL_CARD_GRANTS.pop(RCS_EXECUTOR_MODULE, None)


def _u_shape_fixture() -> tuple[np.ndarray, np.ndarray]:
    """Quadratic (U-shape) truth on a uniform dose axis, fixed seed."""

    rng = np.random.RandomState(0)
    n = 1000
    x = rng.uniform(0.0, 10.0, n)
    y = (x - 5.0) ** 2 + rng.randn(n) * 0.5
    return x, y


def _card_fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Card issuance fixture: U-shape dose plus one linear adjuster."""

    rng = np.random.RandomState(17)
    n = 300
    x = rng.uniform(20.0, 80.0, n)
    cov = rng.randn(n)
    y = 0.02 * (x - 50.0) ** 2 + 0.3 * cov + rng.randn(n) * 0.5
    return x, y, cov


def _card_origin() -> RCSFitResult:
    x, y, cov = _card_fixture()
    return rcs_fit(y, rcs_basis(x, n_knots=4), cov)


def _card_origin_payload() -> dict:
    x, y, cov = _card_fixture()
    fit = rcs_fit(y, rcs_basis(x, n_knots=4), cov)
    fit_output = {k: v for k, v in fit.to_json().items() if k != "basis_sha256"}
    return {
        "kind": "rcs_dose_response_origin/1",
        "inputs": {"x": x.tolist(), "y": y.tolist(), "cov": cov.tolist()},
        "fit": fit_output,
        "nonlinearity": nonlinearity_wald_test(fit).to_json(),
        "curve": predict_curve(fit, 20.0, 80.0, n_grid=50).to_json(),
    }


def _card_origin_digest() -> str:
    return synthetic_origin_sha256(_card_origin_payload())


def test_rcs_card_origin_binds_curve_and_nonlinearity() -> None:
    payload = _card_origin_payload()
    baseline = synthetic_origin_sha256(payload)
    payload["curve"]["predicted"][0] += 0.01
    assert synthetic_origin_sha256(payload) != baseline
    payload = _card_origin_payload()
    payload["nonlinearity"]["statistic"] += 1.0
    assert synthetic_origin_sha256(payload) != baseline


# ---------------------------------------------------------------------------
# RCS basis: exact boundary linearity and knot continuity
# ---------------------------------------------------------------------------


def test_rcs_tails_are_exactly_linear() -> None:
    x, y = _u_shape_fixture()
    basis = rcs_basis(x, n_knots=4)
    fit = rcs_fit(y, basis)
    lower, upper = basis.knots[0], basis.knots[-1]
    span = upper - lower
    for grid_lo, grid_hi in (
        (upper, upper + 0.5 * span),
        (lower - 0.5 * span, lower),
    ):
        curve = predict_curve(fit, grid_lo, grid_hi, n_grid=50)
        second = np.diff(np.asarray(curve.predicted), n=2)
        assert np.all(np.isfinite(second))
        assert float(np.max(np.abs(second))) < 1e-8


def test_rcs_basis_continuous_at_knots() -> None:
    x, _ = _u_shape_fixture()
    basis = rcs_basis(x, n_knots=5)
    knots = np.asarray(basis.knots)
    for knot in basis.knots:
        below = rcs_mod._rcs_columns(np.array([knot - 1e-7]), knots)
        above = rcs_mod._rcs_columns(np.array([knot + 1e-7]), knots)
        at_knot = rcs_mod._rcs_columns(np.array([knot]), knots)
        assert float(np.max(np.abs(below - above))) < 1e-5
        assert float(np.max(np.abs(below - at_knot))) < 1e-5


def test_rcs_explicit_knots_reproduce_quantile_knots() -> None:
    x, _ = _u_shape_fixture()
    auto = rcs_basis(x, n_knots=4)
    assert auto.knot_source == "quantile"
    manual = rcs_basis(x, knots=list(auto.knots))
    assert manual.knot_source == "explicit"
    assert manual.knots == auto.knots
    assert manual.to_json()["matrix"] == auto.to_json()["matrix"]


def test_rcs_cross_check_against_patsy_cr() -> None:
    """Span equality vs ``patsy.cr`` at identical knots (review finding).

    ``patsy.cr`` documents a natural cubic spline basis, so at identical
    knots both bases span the same space even though the parameterization
    differs: OLS fits on either basis must produce numerically identical
    fitted values. Threshold 1e-9 keeps two orders of magnitude of margin
    over the measured ~1e-14 residuals on this fixture/BLAS.
    """

    from patsy import dmatrix

    x, y = _u_shape_fixture()
    basis = rcs_basis(x, n_knots=4)
    knots = list(basis.knots)
    # Harrell knots include both boundaries; patsy cr takes interior knots
    # plus explicit exterior bounds. Without matching bounds patsy anchors
    # linearity at the data extremes instead of t_1/t_k -- a different
    # space, not an implementation disagreement.
    patsy_matrix = dmatrix(
        "cr(x, knots=inner, lower_bound=lo, upper_bound=hi)-1",
        {"x": x, "inner": knots[1:-1], "lo": knots[0], "hi": knots[-1]},
        return_type="dataframe",
    )
    patsy_design = np.column_stack(
        [np.ones(x.shape[0]), np.asarray(patsy_matrix, dtype=float)]
    )
    patsy_coef, _, _, _ = np.linalg.lstsq(patsy_design, y, rcond=None)
    patsy_fitted = patsy_design @ patsy_coef
    ours_design = np.column_stack(
        [np.ones(x.shape[0]), basis.as_array()]
    )
    ours_coef, _, _, _ = np.linalg.lstsq(ours_design, y, rcond=None)
    ours_fitted = ours_design @ ours_coef
    assert float(np.max(np.abs(patsy_fitted - ours_fitted))) < 1e-9


# ---------------------------------------------------------------------------
# Recovery of a known nonlinear DGP; Wald nonlinearity test
# ---------------------------------------------------------------------------


def test_rcs_recovers_u_shape_better_than_linear() -> None:
    x, y = _u_shape_fixture()
    fit = rcs_fit(y, rcs_basis(x, n_knots=4))
    linear_design = np.column_stack([np.ones(x.shape[0]), x])
    linear_coef, _, _, _ = np.linalg.lstsq(linear_design, y, rcond=None)
    linear_rss = float(np.sum((y - linear_design @ linear_coef) ** 2))
    assert fit.rss is not None
    assert fit.rss < 0.5 * linear_rss
    linear_aic = (
        x.shape[0] * np.log(linear_rss / x.shape[0]) + 2.0 * linear_design.shape[1]
    )
    assert fit.aic < linear_aic - 10.0
    wald = nonlinearity_wald_test(fit)
    assert wald.df == 2
    assert tuple(wald.nonlinear_terms) == ("s1", "s2")
    assert wald.p_value < 1e-6


def test_rcs_wald_does_not_reject_linear_truth() -> None:
    rng = np.random.RandomState(1)
    n = 1000
    x = rng.uniform(0.0, 10.0, n)
    y = 2.0 * x + rng.randn(n) * 0.5
    fit = rcs_fit(y, rcs_basis(x, n_knots=4))
    assert nonlinearity_wald_test(fit).p_value > 0.05


def test_rcs_wald_does_not_treat_sex_covariate_as_spline_term() -> None:
    rng = np.random.RandomState(4)
    x = rng.uniform(0.0, 10.0, 400)
    sex = rng.randint(0, 2, 400)
    y = (x - 5.0) ** 2 + 8.0 * sex + rng.normal(0.0, 0.5, 400)
    fit = rcs_fit(
        y,
        rcs_basis(x, n_knots=4),
        covariates=sex.reshape(-1, 1),
        covariate_names=["sex"],
    )

    result = nonlinearity_wald_test(fit)

    assert result.nonlinear_terms == ("s1", "s2")
    assert result.df == 2


def test_rcs_binomial_path_and_probability_curve() -> None:
    rng = np.random.RandomState(17)
    n = 400
    x = rng.uniform(20.0, 80.0, n)
    linear = -3.0 + 0.004 * (x - 50.0) ** 2
    y = (rng.rand(n) < 1.0 / (1.0 + np.exp(-linear))).astype(float)
    fit = rcs_fit(y, rcs_basis(x, n_knots=4), family="binomial")
    assert fit.rss is None
    assert fit.loglik is not None and np.isfinite(fit.aic)
    assert nonlinearity_wald_test(fit).p_value < 1e-6
    curve = predict_curve(fit, 20.0, 80.0, n_grid=50)
    assert curve.response == "probability"
    predicted = np.asarray(curve.predicted)
    assert bool(np.all((predicted >= 0.0) & (predicted <= 1.0)))
    assert bool(np.all(np.asarray(curve.lower) <= predicted))
    assert bool(np.all(predicted <= np.asarray(curve.upper)))
    assert len(curve.grid) == 50


def test_rcs_curve_intervals_cover_and_grid_is_equidistant() -> None:
    x, y = _u_shape_fixture()
    fit = rcs_fit(y, rcs_basis(x, n_knots=4))
    curve = predict_curve(fit, 0.0, 10.0, n_grid=101, level=0.95)
    grid = np.asarray(curve.grid)
    assert len(grid) == 101
    np.testing.assert_allclose(np.diff(grid), 0.1, rtol=0, atol=1e-12)
    predicted = np.asarray(curve.predicted)
    assert bool(np.all(np.asarray(curve.lower) <= predicted))
    assert bool(np.all(predicted <= np.asarray(curve.upper)))
    assert bool(np.all(np.asarray(curve.std_errors) >= 0.0))


# ---------------------------------------------------------------------------
# Determinism + fail-closed
# ---------------------------------------------------------------------------


def test_rcs_digest_stable_across_reruns() -> None:
    first = _card_origin()
    second = _card_origin()
    assert first.to_json() == second.to_json()
    assert rcs_sha256(first) == rcs_sha256(second)
    assert rcs_sha256(first) == canonical_sha256(first.to_json())


def test_rcs_fail_closed() -> None:
    x, y = _u_shape_fixture()
    basis = rcs_basis(x, n_knots=4)
    matrix = basis.as_array()
    good_knots = list(basis.knots)
    with pytest.raises(RCSError):
        rcs_basis(x, knots=[1.0, 1.0, 5.0, 9.0])
    with pytest.raises(RCSError):
        rcs_basis(x, knots=[5.0, 3.0, 6.0, 9.0])
    with pytest.raises(RCSError):
        rcs_basis(x, knots=[3.0, 6.0])
    with pytest.raises(RCSError):
        rcs_basis(x, knots=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    with pytest.raises(RCSError):
        rcs_basis(x, knots=[-5.0, 3.0, 6.0, 9.0])
    with pytest.raises(RCSError):
        rcs_basis(x, knots=[1.0, 3.0, 6.0, 50.0])
    with pytest.raises(RCSError):
        rcs_basis(x, n_knots=2)
    with pytest.raises(RCSError):
        rcs_basis(x, n_knots=8)
    bad_x = np.array(x, dtype=float)
    bad_x[0] = np.nan
    with pytest.raises(RCSError):
        rcs_basis(bad_x, n_knots=4)
    bad_y = np.array(y, dtype=float)
    bad_y[0] = np.inf
    with pytest.raises(RCSError):
        rcs_fit(bad_y, basis)
    with pytest.raises(RCSError):
        rcs_fit(y[:-1], basis)
    with pytest.raises(RCSError):
        rcs_fit(y, matrix)
    with pytest.raises(RCSError):
        rcs_fit(y, matrix, knots=good_knots[:-1])
    with pytest.raises(RCSError):
        rcs_fit(y, matrix, knots=[1.0, 1.0, 5.0, 9.0])
    with pytest.raises(RCSError):
        rcs_fit(y, basis, knots=good_knots)
    with pytest.raises(RCSError):
        rcs_fit(y, basis, family="poisson")
    with pytest.raises(RCSError):
        rcs_fit(y, basis, covariates=np.asarray(x))
    with pytest.raises(RCSError):
        rcs_fit(y, basis, covariates=np.column_stack([x, x]))
    with pytest.raises(RCSError):
        rcs_fit(y, basis, family="binomial")
    single_class = np.zeros(50)
    with pytest.raises(RCSError):
        rcs_fit(single_class, rcs_basis(x[:50], n_knots=3), family="binomial")
    tiny = rcs_basis(x[:3], n_knots=3)
    with pytest.raises(RCSError):
        rcs_fit(y[:3], tiny)
    fit = rcs_fit(y, basis)
    with pytest.raises(RCSError):
        predict_curve(fit, 5.0, 5.0)
    with pytest.raises(RCSError):
        predict_curve(fit, 0.0, 10.0, n_grid=1)
    with pytest.raises(RCSError):
        predict_curve(fit, 0.0, 10.0, level=1.0)
    with pytest.raises(TypeError):
        nonlinearity_wald_test(object())
    with pytest.raises(TypeError):
        predict_curve(object(), 0.0, 1.0)
    with pytest.raises(TypeError):
        rcs_mod.result_sha256(object())
    assert isinstance(basis, RCSBasis)


# ---------------------------------------------------------------------------
# Tool Card: RCS dose-response shape analysis
# ---------------------------------------------------------------------------


def _rcs_card(origin_digest: str) -> ToolCard:
    return ToolCard.model_validate(
        {
            "schema_version": "easyicu.tool_card/1",
            "tool_name": "rcs_dose_response",
            "tool_version": rcs_mod.TOOL_VERSION,
            "method_meaning": (
                "restricted cubic spline dose-response: Harrell "
                "natural-spline basis with exactly linear tails, OLS "
                "gaussian fit, joint Wald test of the nonlinear terms, "
                "and an equidistant curve with delta-method intervals"
            ),
            "inputs": [
                {"name": "x", "value_kind": "float_vector",
                 "description": "continuous exposure vector"},
                {"name": "y", "value_kind": "float_vector",
                 "description": "numeric outcome vector aligned to x"},
                {"name": "covariates", "value_kind": "float_matrix",
                 "required": False,
                 "description": "optional linear adjusters aligned to x"},
            ],
            "outputs": [
                {"name": "coefficients", "value_kind": "float_list"},
                {"name": "nonlinearity_p_value", "value_kind": "float"},
                {"name": "curve_predicted", "value_kind": "float_list"},
                {"name": "curve_interval", "value_kind": "float_list"},
            ],
            "population_assumption": (
                "complete-case numeric exposure/outcome; origin run on a "
                "synthetic U-shape fixture (seed 17, n=300, uniform dose "
                "20..80 plus one gaussian adjuster, 4 quantile knots)"
            ),
            "timing_assumption": (
                "retrospective analysis on a frozen data extract; no "
                "observation-window or real-time constraint"
            ),
            "origin_output_sha256": origin_digest,
            "validation_evidence": [
                {
                    "evidence_id": "rcs-sandbox-run-1",
                    "kind": "sandbox_run",
                    "passed": True,
                    "output_sha256": origin_digest,
                    "note": (
                        "origin run: gaussian RCS, 4 quantile knots, one "
                        "linear adjuster on the U-shape fixture; joint "
                        "Wald p < 1e-6 against linearity"
                    ),
                },
                {
                    "evidence_id": "rcs-repro-1",
                    "kind": "independent_reproduction",
                    "passed": True,
                    "output_sha256": origin_digest,
                    "note": (
                        "deterministic rerun on identical fixed data with "
                        "the kernel seed configuration pinned "
                        "(random_state=0) and a different process-global "
                        "RNG state (numpy seed 999 vs 17); output digest "
                        "byte-identical. NOT a fresh data draw and NOT a "
                        "different knot placement."
                    ),
                },
                {
                    "evidence_id": "rcs-applicability-1",
                    "kind": "applicability_check",
                    "passed": True,
                    "output_sha256": origin_digest,
                    "note": (
                        "intended reuse is continuous-exposure "
                        "dose-response shape analysis on complete-case "
                        "vectors, inside the card population; no clinical "
                        "transport claimed"
                    ),
                },
            ],
        },
        strict=True,
    )


def _build_rcs_envelope(origin_digest: str) -> dict:
    card = _rcs_card(origin_digest)
    decision = decide_tool_promotion(
        card=card,
        reproductions=[
            ReproductionAttempt.model_validate(
                {
                    "attempt_id": "rcs-repro-1",
                    "output_sha256": origin_digest,
                    "passed": True,
                    "independent": True,
                },
                strict=True,
            )
        ],
        executor_module=RCS_EXECUTOR_MODULE,
        applicability=ApplicabilityAttestation.model_validate(
            {
                "intended_population": (
                    "complete-case continuous exposures for "
                    "dose-response shape analysis"
                ),
                "intended_timing": (
                    "retrospective analysis on a frozen extract"
                ),
                "covered": True,
                "reason": (
                    "intended reuse matches the card population and "
                    "timing exactly; synthetic origin bounds the claim "
                    "to dose-response mechanics"
                ),
            },
            strict=True,
        ),
    )
    return {
        "schema_version": "easyicu.tool_card_grant_file/1",
        "executor_module": RCS_EXECUTOR_MODULE,
        "issued_note": (
            "Issued 2026-09-18 on a synthetic U-shape fixture; origin "
            "bounds the claim to dose-response mechanics (see card "
            "population assumption). Reproduction is deterministic "
            "rerun, not fresh data; knot placement must be reported on "
            "reuse."
        ),
        "card": card.model_dump(mode="json"),
        "decision": decision.model_dump(mode="json"),
    }


def test_rcs_tool_card_promotion_and_grant() -> None:
    _clear_grant()
    try:
        origin = _card_origin()
        origin_digest = _card_origin_digest()
        assert nonlinearity_wald_test(origin).p_value < 1e-6

        np.random.seed(999)
        try:
            rerun = _card_origin()
            rerun_digest = _card_origin_digest()
        finally:
            np.random.seed()
        assert rcs_sha256(rerun) == rcs_sha256(origin)
        assert rerun_digest == origin_digest

        card = _rcs_card(origin_digest)
        assert tool_card_completeness_issues(card) == []

        envelope = _build_rcs_envelope(origin_digest)
        decision = decide_tool_promotion(
            card=card,
            reproductions=[
                ReproductionAttempt.model_validate(
                    {
                        "attempt_id": "rcs-repro-1",
                        "output_sha256": rerun_digest,
                        "passed": True,
                        "independent": True,
                    },
                    strict=True,
                )
            ],
            executor_module=RCS_EXECUTOR_MODULE,
            applicability=ApplicabilityAttestation.model_validate(
                {
                    "intended_population": (
                        "complete-case continuous exposures for "
                        "dose-response shape analysis"
                    ),
                    "intended_timing": (
                        "retrospective analysis on a frozen extract"
                    ),
                    "covered": True,
                    "reason": (
                        "intended reuse matches the card population and "
                        "timing exactly; synthetic origin bounds the "
                        "claim to dose-response mechanics"
                    ),
                },
                strict=True,
            ),
        )
        assert decision.granted_verified_tool is True
        assert decision.allowed_identity == "verified_tool"
        assert decision.card_sha256 == tool_card_sha256(card)
        assert (
            envelope["decision"]["card_sha256"] == decision.card_sha256
        )
        require_verified_tool(decision)

        record = registry_module.register_tool_card_grant(
            executor_module=RCS_EXECUTOR_MODULE,
            card=card,
            decision=decision,
        )
        assert record["tool_name"] == "rcs_dose_response"
        assert record["tool_version"] == rcs_mod.TOOL_VERSION
        assert record["card_sha256"] == tool_card_sha256(card)
        assert (
            registry_module.granted_tool_identity(RCS_EXECUTOR_MODULE)
            == "verified_tool"
        )
    finally:
        _clear_grant()


def _tool_cards_root() -> Path:
    return (
        Path(registry_module.__file__).resolve().parent.parent
        / "methods"
        / "tool_cards"
    )


def test_rcs_checked_in_envelope_loads_and_origin_reproduces(tmp_path) -> None:
    saved = dict(registry_module._TOOL_CARD_GRANTS)
    saved_flag = registry_module._CHECKED_IN_TOOL_CARDS_LOADED
    registry_module._TOOL_CARD_GRANTS.clear()
    registry_module._CHECKED_IN_TOOL_CARDS_LOADED = False
    try:
        origin_digest = _card_origin_digest()
        envelope = _build_rcs_envelope(origin_digest)

        staged = tmp_path / "cards"
        shutil.copytree(_tool_cards_root(), staged)
        (staged / "rcs_dose_response.card.json").write_text(
            json.dumps(envelope, sort_keys=True, indent=2), encoding="utf-8"
        )
        records = registry_module.load_checked_in_tool_cards(directory=staged)
        assert records[RCS_EXECUTOR_MODULE]["tool_name"] == "rcs_dose_response"
        assert (
            registry_module.granted_tool_identity(RCS_EXECUTOR_MODULE)
            == "verified_tool"
        )

        checked_in = json.loads(
            (_tool_cards_root() / "rcs_dose_response.card.json").read_text(
                encoding="utf-8"
            )
        )
        assert (
            checked_in["card"]["origin_output_sha256"] == origin_digest
        )
        live_card = ToolCard.model_validate(checked_in["card"], strict=True)
        assert (
            checked_in["decision"]["card_sha256"] == tool_card_sha256(live_card)
        )
    finally:
        registry_module._TOOL_CARD_GRANTS.clear()
        registry_module._TOOL_CARD_GRANTS.update(saved)
        registry_module._CHECKED_IN_TOOL_CARDS_LOADED = saved_flag
