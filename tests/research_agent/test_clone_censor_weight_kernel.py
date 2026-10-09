"""A clone-censor-weight trial recovers the risks a synthetic truth fixes.

The synthetic stays follow a known process: baseline severity raises both the
hazard of starting the treatment in the grace period and the hazard of death,
healthier stays leave the ICU sooner, a death in the grace period does not
depend on a start, and after the grace period the treatment lowers the death
hazard while untreated stays may still start.  The risk under each strategy
is computed from that process directly.  Within one grace-period hour a start
comes first, then an ICU exit, then a death, so the censoring the estimator
places at the start of an hour precedes that hour's deaths.  No benchmark item
or patient row is used.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pytest

from easyicu.research_agent.contracts.target_trial_design import (
    MAX_GRACE_PERIOD_HOURS,
    MIN_TIME_ZERO_HOURS,
    TARGET_TRIAL_HOST_POLICY,
    TARGET_TRIAL_STOP_REASONS,
    TARGET_TRIAL_STOP_THRESHOLDS,
    WEIGHT_MODEL_NOT_ESTIMABLE_CAUSES,
)
from easyicu.research_agent.methods import clone_censor_weight as cw
from easyicu.research_agent.methods.clone_censor_weight import (
    DEFER,
    INITIATE,
    TIME_ZERO_EXCLUSIONS,
    TIME_ZERO_INCLUSIONS,
    CloneCensorWeightError,
    TrialTiming,
    bootstrap_clone_censor_weight,
    estimate_clone_censor_weight,
    hour_terms,
    icu_exit_model_form,
    resample_trial_units,
    time_zero_eligibility,
    trial_course,
    trial_stays,
)
from easyicu.research_agent.methods.clone_censor_weight_diagnostics import (
    adherence_summaries,
    covariate_balance,
    grace_icu_exit_share,
    late_start_count,
    positivity_summary,
    risk_ratio_e_value,
    weight_summaries,
)
from easyicu.research_agent.methods.sensitivity import compute_e_value
from easyicu.research_agent.planning import target_trial_compile

T0 = 2
GRACE = 6
HORIZON_DAYS = 28
TIMING = TrialTiming(
    time_zero_hours=T0, grace_period_hours=GRACE, horizon_hours=HORIZON_DAYS * 24
)
EFFECT = math.log(0.7)


def _expit(values):
    return 1.0 / (1.0 + np.exp(-values))


def _simulate(
    n: int,
    seed: int,
    *,
    confounded: bool = True,
    exits: bool = True,
    grace_death: float = -5.6,
) -> dict:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = (rng.random(n) < 0.4).astype(float)
    strength = 1.0 if confounded else 0.0
    onset = np.full(n, np.nan)
    death = np.full(n, np.nan)
    exit_ = np.full(n, np.nan)
    started = np.zeros(n, dtype=bool)
    alive = np.ones(n, dtype=bool)
    in_icu = np.ones(n, dtype=bool)
    for hour in range(GRACE):
        at_risk = alive & in_icu & ~started
        hazard = _expit(-2.6 + 0.15 * hour + strength * (0.9 * x1 + 0.6 * x2))
        starts = at_risk & (rng.random(n) < hazard)
        onset[starts] = T0 + hour + 0.1
        started |= starts
        if exits:
            leaving = _expit(-4.2 - strength * 0.7 * x1)
            leaves = alive & in_icu & ~started & (rng.random(n) < leaving)
            exit_[leaves] = T0 + hour + 0.4
            in_icu &= ~leaves
        dies = alive & (rng.random(n) < _expit(grace_death + 0.8 * x1 + 0.4 * x2))
        death[dies] = T0 + hour + 0.7
        alive &= ~dies
    stays_on = np.isnan(exit_)
    exit_[stays_on] = T0 + GRACE + rng.exponential(48.0, size=int(stays_on.sum()))
    survivors = np.flatnonzero(alive)
    hazard0 = np.exp(-8.3 + 0.7 * x1 + 0.4 * x2)[survivors]
    later_start = np.where(
        started[survivors],
        0.0,
        rng.exponential(1.0 / np.exp(-7.0 + 0.5 * x1[survivors])),
    )
    draw = rng.exponential(1.0, size=survivors.shape[0])
    before = hazard0 * later_start
    time_to_death = np.where(
        draw < before,
        draw / hazard0,
        later_start + (draw - before) / (hazard0 * math.exp(EFFECT)),
    )
    follow_up = HORIZON_DAYS * 24 - (T0 + GRACE)
    dead = time_to_death <= follow_up
    death[survivors[dead]] = T0 + GRACE + time_to_death[dead]
    return {"x1": x1, "x2": x2, "onset": onset, "death": death, "exit": exit_}


def _truth(*, grace_death: float = -5.6) -> tuple[float, float]:
    """Each strategy's risk by the horizon, integrated over the covariates."""

    rng = np.random.default_rng(20261009)
    x1 = rng.normal(size=1_000_000)
    x2 = (rng.random(1_000_000) < 0.4).astype(float)
    alive_after_grace = (1.0 - _expit(grace_death + 0.8 * x1 + 0.4 * x2)) ** GRACE
    hazard0 = np.exp(-8.3 + 0.7 * x1 + 0.4 * x2)
    later = np.exp(-7.0 + 0.5 * x1)
    ratio = math.exp(EFFECT)
    tau = HORIZON_DAYS * 24 - (T0 + GRACE)
    treated = np.exp(-hazard0 * ratio * tau)
    rate = hazard0 + later - hazard0 * ratio
    untreated = (
        np.exp(-(hazard0 + later) * tau)
        + later * np.exp(-hazard0 * ratio * tau) * (1.0 - np.exp(-rate * tau)) / rate
    )
    return (
        float(1.0 - (alive_after_grace * treated).mean()),
        float(1.0 - (alive_after_grace * untreated).mean()),
    )


def _stays(sim: dict, *, groups=None, endpoint=None, extra=None, window=None):
    columns = [sim["x1"], sim["x2"]] + ([] if extra is None else [extra])
    names = ["x1", "x2"] + ([] if extra is None else ["x3"])
    n = sim["x1"].shape[0]
    return trial_stays(
        stay_ids=np.arange(n),
        group_ids=np.arange(n) if groups is None else groups,
        onset_hours=sim["onset"],
        death_hours=sim["death"],
        icu_exit_hours=sim["exit"],
        endpoint_observed=np.ones(n, dtype=bool) if endpoint is None else endpoint,
        covariates=np.column_stack(columns),
        covariate_names=names,
        onset_window_hours=(0.0, T0 + GRACE) if window is None else window,
    )


def _tiny(onset, death, exit_, *, observed=None, follows=None):
    n = len(onset)
    return trial_stays(
        stay_ids=[f"s{i}" for i in range(n)],
        group_ids=[f"s{i}" for i in range(n)],
        onset_hours=onset,
        death_hours=death,
        icu_exit_hours=exit_,
        endpoint_observed=np.ones(n, dtype=bool) if observed is None else observed,
        covariates=np.zeros((n, 0)),
        covariate_names=(),
        onset_window_hours=(0.0, 24.0),
        death_follows_icu_exit=None if follows is None else np.asarray(follows),
    )


NAN = float("nan")


# -- time zero, hours and their order ----------------------------------------


def test_time_zero_rules_exclude_in_protocol_order() -> None:
    stays = _tiny(
        onset=[NAN, NAN, NAN, NAN, NAN, NAN, T0 - 0.5, 0.0, T0, T0 + 0.5],
        death=[NAN, NAN, T0, T0 - 1, NAN, NAN, NAN, NAN, NAN, NAN],
        exit_=[50, 50, 50, 50, T0, NAN, 50, 50, 50, 50],
        observed=np.array([True, False] + [True] * 8),
    )

    report = time_zero_eligibility(stays, TIMING)

    assert tuple(step.name for step in report.steps) == (
        TIME_ZERO_INCLUSIONS + TIME_ZERO_EXCLUSIONS
    )
    assert [(step.name, step.kind, step.n_removed) for step in report.steps] == [
        ("endpoint_observed", "inclusion", 1),
        ("alive_at_time_zero", "inclusion", 2),
        ("in_icu_at_time_zero", "inclusion", 2),
        ("treatment_started_before_time_zero", "exclusion", 2),
    ]
    # A start at time zero itself is in the grace period, not prevalent.
    assert report.analysis_positions == (0, 8, 9)


def test_the_grace_period_is_closed_at_time_zero_and_open_at_its_end() -> None:
    stays = _tiny(
        onset=[T0, T0 + 1.0, T0 + 1.0001, T0 + GRACE - 0.01, T0 + GRACE, NAN],
        death=[NAN] * 6,
        exit_=[50.0] * 6,
    )

    course = trial_course(stays, TIMING)

    assert course.started.tolist() == [True, True, True, True, False, False]
    assert course.start_hour.tolist() == [0, 0, 1, GRACE - 1, -1, -1]
    assert course.last_hour_at_risk.tolist() == [
        0,
        0,
        1,
        GRACE - 1,
        GRACE - 1,
        GRACE - 1,
    ]


def test_within_an_hour_a_start_comes_before_a_death_and_a_death_before_an_exit() -> (
    None
):
    stays = _tiny(
        onset=[T0 + 2.5, T0 + 2.6, T0 + 3.2, T0 + 3.5, NAN, NAN, NAN],
        death=[T0 + 2.5, T0 + 2.3, NAN, NAN, T0 + 4.4, T0 + 4.6, T0 + GRACE],
        exit_=[50.0, 50.0, T0 + 3.2, T0 + 3.1, T0 + 4.4, T0 + 4.2, 50.0],
    )

    course = trial_course(stays, TIMING)

    assert course.started.tolist() == [True, False, True, False, False, False, False]
    assert course.leaves_icu_before_start.tolist() == [
        False,
        False,
        False,
        True,
        False,
        True,
        False,
    ]
    assert course.exit_hour.tolist() == [-1, -1, -1, 3, -1, 4, -1]
    assert course.start_after_death_or_exit.tolist() == [
        False,
        True,
        False,
        True,
        False,
        False,
        False,
    ]
    assert course.death_in_grace.tolist() == [
        True,
        True,
        False,
        False,
        True,
        True,
        True,
    ]
    assert course.death_hour.tolist() == [2, 2, -1, -1, 4, 4, GRACE - 1]


def test_a_death_known_to_follow_the_icu_exit_comes_after_an_exit_at_its_time() -> None:
    # A death after hospital discharge known only by its day is placed at the
    # ICU exit when the day precedes it: the stay left the ICU, then died, so
    # the starting strategy censors it at the exit rather than counting it.
    stays = _tiny(
        onset=[NAN, NAN, T0 + 1.5, NAN],
        death=[T0 + 4.4, T0 + 4.4, T0 + 4.4, T0 + GRACE + 3.0],
        exit_=[T0 + 4.4, T0 + 4.4, T0 + 4.4, T0 + GRACE + 3.0],
        follows=[False, True, True, True],
    )

    course = trial_course(stays, TIMING)

    assert course.leaves_icu_before_start.tolist() == [False, True, False, False]
    assert course.exit_hour.tolist() == [-1, 4, -1, -1]
    assert course.death_in_grace.tolist() == [True, True, True, False]
    assert course.started.tolist() == [False, False, True, False]
    assert stays.take(np.array([1, 1])).death_follows_icu_exit.tolist() == [True, True]


@pytest.mark.parametrize(
    ("death", "exit_", "follows"),
    [
        ([T0 + 2.0], [T0 + 3.0], [True]),
        ([NAN], [T0 + 3.0], [True]),
        ([T0 + 3.0], [T0 + 3.0], [1]),
        ([T0 + 3.0], [T0 + 3.0], [True, True]),
    ],
    ids=["precedes_its_exit", "has_no_time", "not_a_boolean", "not_one_per_stay"],
)
def test_a_death_known_to_follow_the_icu_exit_must_follow_it(
    death, exit_, follows
) -> None:
    with pytest.raises(CloneCensorWeightError) as raised:
        _tiny(onset=[NAN], death=death, exit_=exit_, follows=follows)
    assert raised.value.code == "ccw_input_invalid"


def test_a_death_known_to_follow_an_unknown_icu_exit_leaves_the_stay_ineligible() -> (
    None
):
    stays = _tiny(
        onset=[NAN, NAN],
        death=[T0 + 3.0, T0 + 3.0],
        exit_=[NAN, T0 + 3.0],
        follows=[True, True],
    )

    assert time_zero_eligibility(stays, TIMING).analysis_positions == (1,)


def test_a_stay_that_was_not_eligible_cannot_reach_the_course() -> None:
    with pytest.raises(CloneCensorWeightError) as raised:
        trial_course(_tiny(onset=[T0 - 0.5], death=[NAN], exit_=[50.0]), TIMING)
    assert raised.value.code == "ccw_input_invalid"


# -- clones, censoring and weights -------------------------------------------

_PROBES = {
    # onset, death, exit
    "dies_before_starting": (NAN, T0 + 2.5, 50.0),
    "starts_then_dies": (T0 + 1.5, T0 + 3.5, 50.0),
    "starts": (T0 + 2.4, NAN, 50.0),
    "never_starts": (NAN, NAN, 50.0),
    "leaves_before_starting": (NAN, NAN, T0 + 1.3),
}


@pytest.fixture(scope="module")
def probed():
    sim = _simulate(3000, 11)
    for key, value in zip(("onset", "death", "exit"), zip(*_PROBES.values())):
        sim[key] = np.concatenate([sim[key], np.asarray(value)])
    sim["x1"] = np.concatenate([sim["x1"], np.zeros(len(_PROBES))])
    sim["x2"] = np.concatenate([sim["x2"], np.zeros(len(_PROBES))])
    estimate = estimate_clone_censor_weight(_stays(sim), TIMING)
    positions = {
        name: int(np.flatnonzero(estimate.eligible.stay_ids == 3000 + index)[0])
        for index, name in enumerate(_PROBES)
    }
    return estimate, positions


def _rows(arm, position):
    keep = arm.row_stay == position
    return list(
        zip(
            arm.row_entry[keep].tolist(),
            arm.row_stop[keep].tolist(),
            arm.row_event[keep].tolist(),
        )
    )


def test_a_death_in_the_grace_period_before_a_start_counts_in_both_arms(probed) -> None:
    estimate, at = probed
    for arm in (INITIATE, DEFER):
        rows = _rows(estimate.arms[arm], at["dies_before_starting"])
        assert rows[-1] == (2.0, 2.5, True)
        assert len(rows) == 3
        assert estimate.arms[arm].uncensored_at_grace_end[at["dies_before_starting"]]
    # A death after a start is the initiating strategy's only.
    assert _rows(estimate.arms[INITIATE], at["starts_then_dies"])[-1] == (
        3.0,
        3.5,
        True,
    )
    assert _rows(estimate.arms[DEFER], at["starts_then_dies"]) == [(0.0, 1.0, False)]


def test_each_clone_is_censored_where_its_strategy_says(probed) -> None:
    estimate, at = probed
    initiate, defer = estimate.arms[INITIATE], estimate.arms[DEFER]
    follow_up = float(TIMING.follow_up_hours)
    # A start in hour 2 censors the deferring clone when that hour begins.
    assert _rows(defer, at["starts"])[-1] == (1.0, 2.0, False)
    assert _rows(initiate, at["starts"])[-1] == (GRACE, follow_up, False)
    # No start by the end of the grace period censors the initiating clone then.
    assert _rows(initiate, at["never_starts"])[-1] == (GRACE - 1, GRACE, False)
    assert _rows(defer, at["never_starts"])[-1] == (GRACE, follow_up, False)
    # Leaving the ICU before a start censors only the initiating clone.
    assert _rows(initiate, at["leaves_before_starting"]) == [(0.0, 1.0, False)]
    assert _rows(defer, at["leaves_before_starting"])[-1] == (GRACE, follow_up, False)


def _common_start_probability(estimate) -> float:
    params = np.asarray(estimate.initiation_numerator.params)
    grid = hour_terms(np.arange(GRACE), GRACE)
    common = _expit(params[0] + grid @ params[1:])
    return 1.0 - float(np.prod(1.0 - common))


def test_weights_stay_constant_after_the_grace_period(probed) -> None:
    estimate, _ = probed
    common_start = _common_start_probability(estimate)
    for name in (INITIATE, DEFER):
        arm = estimate.arms[name]
        post = arm.row_entry == GRACE
        last_hour = arm.row_entry == GRACE - 1
        assert np.array_equal(
            np.sort(arm.row_stay[post]), np.flatnonzero(arm.followed_past_grace)
        )
        end_of_grace = dict(zip(arm.row_stay[last_hour], arm.row_weight[last_hour]))
        carried = np.array([end_of_grace[stay] for stay in arm.row_stay[post]])
        factor = (
            common_start / estimate.start_probability[arm.row_stay[post]]
            if name == INITIATE
            else 1.0
        )
        assert np.allclose(arm.row_weight[post], carried * factor, rtol=1e-12)
        assert np.allclose(
            arm.grace_end_weight[arm.row_stay[post]], arm.row_weight[post]
        )


def test_the_weights_are_stabilised() -> None:
    estimate = estimate_clone_censor_weight(_stays(_simulate(12000, 3)), TIMING)
    summaries = {
        (item.arm, item.truncated): item for item in weight_summaries(estimate)
    }
    for arm in (INITIATE, DEFER):
        before = summaries[(arm, False)]
        after = summaries[(arm, True)]
        fit = estimate.arms[arm]
        raw = fit.grace_end_weight[fit.followed_past_grace]
        assert before.n == after.n == fit.n_followed_past_grace
        assert (before.minimum, before.maximum) == (raw.min(), raw.max())
        assert before.ess == pytest.approx(raw.sum() ** 2 / np.square(raw).sum())
        assert (after.minimum, after.maximum) == fit.truncation_bounds
        assert 0.8 < before.mean < 1.2
        assert after.n_clipped_high > 0 and after.ess > before.ess
        assert before.ess_share > 0.1


@pytest.mark.parametrize(
    ("exits", "stays", "parameters", "form"),
    [
        (0, 1000, 5, "none"),
        (49, 10000, 5, "hour_terms_only"),
        (50, 10000, 5, "hour_terms_and_covariates"),
        (40, 2000, 5, "hour_terms_only"),
        (41, 2000, 5, "hour_terms_and_covariates"),
    ],
)
def test_the_icu_exit_model_form_follows_its_exits_and_their_share(
    exits, stays, parameters, form
) -> None:
    assert icu_exit_model_form(exits, stays, parameters) == form


def test_rare_icu_exits_are_modelled_on_the_hours_alone() -> None:
    sim = _simulate(10000, 900)
    rng = np.random.default_rng(0)
    early = sim["exit"] <= T0 + GRACE
    later = T0 + GRACE + rng.exponential(48.0, 10000)
    sim["exit"] = np.where(early & (rng.random(10000) > 0.05), later, sim["exit"])
    extra = np.column_stack(
        [(rng.random(10000) < rng.uniform(0.03, 0.3)).astype(float) for _ in range(12)]
    )
    stays = trial_stays(
        stay_ids=np.arange(10000),
        group_ids=np.arange(10000),
        onset_hours=sim["onset"],
        death_hours=sim["death"],
        icu_exit_hours=sim["exit"],
        endpoint_observed=np.ones(10000, dtype=bool),
        covariates=np.column_stack([sim["x1"], sim["x2"], extra]),
        covariate_names=["x1", "x2", *(f"b{index}" for index in range(12))],
        onset_window_hours=(0.0, T0 + GRACE),
    )
    full = 1 + 2 + 14
    leaving = int((early & (sim["exit"] <= T0 + GRACE)).sum())
    assert 0 < leaving < 10 * full and leaving <= 0.02 * 10000

    estimate = estimate_clone_censor_weight(stays, TIMING)

    assert estimate.icu_exit_model_form == "hour_terms_only"
    assert estimate.icu_exit_model.terms == ("intercept", "hour", "hour_spline")
    assert estimate.icu_exit_model == estimate.icu_exit_numerator
    assert estimate.initiation_model.terms[-1] == "b11"


def test_each_weight_model_is_fitted_on_the_hours_at_risk() -> None:
    estimate = estimate_clone_censor_weight(_stays(_simulate(3000, 13)), TIMING)
    course = estimate.course
    starting = int((course.last_hour_at_risk + 1).sum())
    leaving = int((course.last_hour_without_start + 1).sum())
    assert (estimate.initiation_model.n_rows, estimate.initiation_model.n_events) == (
        starting,
        int(course.started.sum()),
    )
    assert (estimate.icu_exit_model.n_rows, estimate.icu_exit_model.n_events) == (
        leaving,
        int(course.leaves_icu_before_start.sum()),
    )
    assert estimate.initiation_numerator.n_rows == starting
    assert estimate.icu_exit_numerator.n_rows == leaving
    assert estimate.icu_exit_model_form == "hour_terms_and_covariates"
    assert estimate.initiation_model.terms == (
        "intercept",
        "hour",
        "hour_spline",
        "x1",
        "x2",
    )
    assert estimate.initiation_numerator.terms == ("intercept", "hour", "hour_spline")


def _product_limit(arm, weights, follow_up: float) -> float:
    """Weighted Kaplan-Meier over counting-process rows, written out."""

    survival = 1.0
    times = np.unique(arm.row_stop[arm.row_event])
    for time in times[times <= follow_up]:
        deaths = weights[arm.row_event & (arm.row_stop == time)].sum()
        at_risk = weights[(arm.row_entry < time) & (arm.row_stop >= time)].sum()
        survival *= 1.0 - deaths / at_risk
    return 1.0 - survival


def test_each_arm_risk_is_the_weighted_product_limit_of_its_rows(probed) -> None:
    estimate, _ = probed
    follow_up = TIMING.follow_up_hours
    for name in (INITIATE, DEFER):
        arm = estimate.arms[name]
        low, high = arm.truncation_bounds
        post = arm.grace_end_weight[arm.followed_past_grace]
        assert (low, high) == tuple(np.percentile(post, [1.0, 99.0]))
        assert np.array_equal(
            arm.row_weight_truncated, np.clip(arm.row_weight, low, high)
        )
        assert arm.risk == pytest.approx(
            _product_limit(arm, arm.row_weight, follow_up), abs=1e-10
        )
        assert arm.risk_truncated == pytest.approx(
            _product_limit(arm, arm.row_weight_truncated, follow_up), abs=1e-10
        )
        assert arm.risk_crude == pytest.approx(
            _product_limit(arm, np.ones_like(arm.row_weight), follow_up), abs=1e-10
        )
        assert arm.curve_risk[-1] == pytest.approx(arm.risk, abs=1e-12)
        assert arm.n_events == int(arm.row_event.sum())


def test_balance_compares_uncensored_clones_with_the_eligible_population() -> None:
    estimate = estimate_clone_censor_weight(_stays(_simulate(12000, 5)), TIMING)
    rows = {(row.arm, row.covariate): row for row in covariate_balance(estimate)}
    covariates = estimate.eligible.covariates
    for name in (INITIATE, DEFER):
        arm = estimate.arms[name]
        kept = arm.uncensored_at_grace_end
        # Deaths in the grace period stay in the compared set.
        assert (
            (kept & ~arm.followed_past_grace).sum()
            == (estimate.course.death_in_grace & kept).sum()
            > 0
        )
        weights = arm.grace_end_weight[kept]
        for column, covariate in enumerate(("x1", "x2")):
            values = covariates[:, column]
            sd = values.std(ddof=1)
            row = rows[(name, covariate)]
            assert row.eligible_mean == pytest.approx(values.mean())
            assert row.eligible_sd == pytest.approx(sd)
            weighted = np.average(values[kept], weights=weights)
            assert row.smd_weighted == pytest.approx((weighted - values.mean()) / sd)
            assert row.smd_unweighted == pytest.approx(
                (values[kept].mean() - values.mean()) / sd
            )
        assert abs(rows[(name, "x1")].smd_unweighted) > 0.2
        assert abs(rows[(name, "x1")].smd_weighted) < 0.05


def test_positivity_adherence_and_icu_exit_are_reported_per_stay(probed) -> None:
    estimate, _ = probed
    positivity = positivity_summary(estimate)
    low, high = TARGET_TRIAL_HOST_POLICY["positivity_window"]
    probability = estimate.start_probability
    assert positivity.window == (low, high)
    assert positivity.n_outside == int(
        ((probability <= low) | (probability >= high)).sum()
    )
    assert positivity.n_started + positivity.n_not_started == estimate.eligible.n
    adherence = {item.arm: item for item in adherence_summaries(estimate)}
    course = estimate.course
    assert sum(adherence[DEFER].censored_by_hour) == int(course.started.sum())
    assert sum(adherence[INITIATE].censored_by_hour) == int(
        course.leaves_icu_before_start.sum()
    )
    for name in (INITIATE, DEFER):
        arm = estimate.arms[name]
        item = adherence[name]
        censored = sum(item.censored_by_hour) + item.censored_at_grace_end
        assert (
            censored + item.deaths_in_grace + item.followed_past_grace == arm.n_clones
        )
    assert grace_icu_exit_share(estimate) == pytest.approx(
        course.leaves_icu_before_start.mean()
    )
    assert late_start_count(estimate) == 0


# -- the synthetic truth -------------------------------------------------------


def test_confounded_starts_are_recovered_by_the_weights_and_not_by_the_crude_risks() -> (
    None
):
    truth_initiate, truth_defer = _truth()
    estimates = [
        estimate_clone_censor_weight(_stays(_simulate(12000, seed)), TIMING)
        for seed in range(5)
    ]
    weighted = np.array(
        [[item.arms[INITIATE].risk, item.arms[DEFER].risk] for item in estimates]
    )
    crude = np.array(
        [
            [item.arms[INITIATE].risk_crude, item.arms[DEFER].risk_crude]
            for item in estimates
        ]
    )
    truth = np.array([truth_initiate, truth_defer])
    assert np.all(np.abs(weighted.mean(axis=0) - truth) < 0.01)
    weighted_difference = (weighted[:, 0] - weighted[:, 1]).mean()
    crude_difference = (crude[:, 0] - crude[:, 1]).mean()
    assert abs(weighted_difference - (truth_initiate - truth_defer)) < 0.01
    assert crude_difference - (truth_initiate - truth_defer) > 0.05


def test_randomised_starts_need_no_weights() -> None:
    truth_initiate, truth_defer = _truth()
    for seed in range(3):
        estimate = estimate_clone_censor_weight(
            _stays(_simulate(12000, 100 + seed, confounded=False, exits=False)), TIMING
        )
        for name, truth in ((INITIATE, truth_initiate), (DEFER, truth_defer)):
            arm = estimate.arms[name]
            assert abs(arm.risk - arm.risk_crude) < 0.01
            assert abs(arm.risk - truth) < 0.02


# -- uncertainty ---------------------------------------------------------------


def test_a_resample_draws_whole_units() -> None:
    groups = np.array(["a", "b", "a", "c", "c", "c", "d"], dtype=object)
    rng = np.random.default_rng(0)
    for _ in range(50):
        positions = resample_trial_units(groups, rng)
        drawn = groups[positions]
        for unit in np.unique(groups):
            members = np.flatnonzero(groups == unit)
            counts = [int((positions == member).sum()) for member in members]
            assert len(set(counts)) == 1
            assert int((drawn == unit).sum()) == counts[0] * members.shape[0]
        units_drawn = sum(
            int((positions == np.flatnonzero(groups == unit)[0]).sum())
            for unit in np.unique(groups)
        )
        assert units_drawn == 4


def test_each_resample_refits_both_clones_of_every_drawn_stay_once(monkeypatch) -> None:
    sim = _simulate(1500, 21)
    estimate = estimate_clone_censor_weight(
        _stays(sim, groups=np.arange(1500) // 2), TIMING
    )
    calls = []
    real = cw._estimate_eligible

    def spy(eligible, timing, eligibility, **kwargs):
        result = real(eligible, timing, eligibility, **kwargs)
        calls.append(
            (
                eligible.stay_ids.copy(),
                result.arms[INITIATE].n_clones,
                result.arms[DEFER].n_clones,
                result.arms[INITIATE].risk,
                result.arms[DEFER].risk,
            )
        )
        return result

    monkeypatch.setattr(cw, "_estimate_eligible", spy)
    boot = bootstrap_clone_censor_weight(estimate, resamples=6, seed=7)

    assert len(calls) == 6
    rng = np.random.default_rng(7)
    for (ids, n_initiate, n_defer, risk_initiate, risk_defer), row in zip(
        calls, boot.replicate_risks
    ):
        drawn = estimate.eligible.stay_ids[
            resample_trial_units(estimate.eligible.group_ids, rng)
        ]
        assert np.array_equal(ids, drawn)
        assert n_initiate == n_defer == ids.shape[0]
        assert tuple(row) == (risk_initiate, risk_defer)


def test_the_bootstrap_is_reproducible_and_counts_its_failures(monkeypatch) -> None:
    estimate = estimate_clone_censor_weight(_stays(_simulate(1500, 23)), TIMING)
    first = bootstrap_clone_censor_weight(estimate, resamples=5, seed=1)
    second = bootstrap_clone_censor_weight(estimate, resamples=5, seed=1)
    assert np.array_equal(first.replicate_risks, second.replicate_risks)
    assert first.intervals == second.intervals
    assert (
        first.replicate_risks.shape == first.replicate_risks_truncated.shape == (5, 2)
    )
    low, high = first.intervals.risk_difference
    differences = first.replicate_risks[:, 0] - first.replicate_risks[:, 1]
    assert low == pytest.approx(np.percentile(differences, 2.5))
    assert high == pytest.approx(np.percentile(differences, 97.5))

    real = cw._estimate_eligible
    count = {"n": 0}

    def failing(eligible, timing, eligibility, **kwargs):
        count["n"] += 1
        if count["n"] % 2 == 0:
            raise CloneCensorWeightError(
                "synthetic", code="ccw_weight_model_not_estimable", cause="separation"
            )
        return real(eligible, timing, eligibility, **kwargs)

    monkeypatch.setattr(cw, "_estimate_eligible", failing)
    boot = bootstrap_clone_censor_weight(estimate, resamples=6, seed=1)
    assert boot.failures == {"ccw_weight_model_not_estimable:separation": 3}
    assert boot.failure_share == 0.5
    assert boot.replicate_risks.shape == (3, 2)


def test_the_risk_ratio_e_value_is_the_sensitivity_owners() -> None:
    assert risk_ratio_e_value(0.8, (0.7, 0.9)) == compute_e_value(
        estimate=0.8, ci=(0.7, 0.9), estimate_type="rr"
    )
    assert risk_ratio_e_value(None, None) is None


# -- typed refusals ------------------------------------------------------------


@pytest.mark.parametrize(
    ("time_zero", "grace", "horizon"),
    [
        (0, 6, 672),
        (73, 6, 672),
        (2, 0, 672),
        (2, MAX_GRACE_PERIOD_HOURS + 1, 672),
        (2, 6, 8),
        (2.0, 6, 672),
        (True, 6, 672),
    ],
)
def test_a_timing_outside_the_host_menu_is_refused(time_zero, grace, horizon) -> None:
    with pytest.raises(CloneCensorWeightError) as raised:
        TrialTiming(
            time_zero_hours=time_zero, grace_period_hours=grace, horizon_hours=horizon
        )
    assert raised.value.code == "ccw_timing_invalid"


def test_the_policy_and_the_stop_thresholds_are_fixed_constants() -> None:
    for mapping in (TARGET_TRIAL_HOST_POLICY, TARGET_TRIAL_STOP_THRESHOLDS):
        key = next(iter(mapping))
        with pytest.raises(TypeError):
            mapping[key] = None
    assert TARGET_TRIAL_HOST_POLICY["primary_weights"] == "stabilized_untruncated"
    assert "stabilized_truncated" in TARGET_TRIAL_HOST_POLICY["sensitivity_weights"]
    assert len(set(TARGET_TRIAL_STOP_REASONS)) == len(TARGET_TRIAL_STOP_REASONS)


def test_the_kernel_and_the_compiler_offer_the_same_trial_hours() -> None:
    assert target_trial_compile.TIME_ZERO_MENU_HOURS[0] == MIN_TIME_ZERO_HOURS
    assert target_trial_compile.MAX_GRACE_PERIOD_HOURS == MAX_GRACE_PERIOD_HOURS


@pytest.mark.parametrize(
    "change",
    [
        {"onset_hours": [float("inf"), NAN]},
        {"stay_ids": ["a", "a"]},
        {"group_ids": ["a", None]},
        {"endpoint_observed": [1, 1]},
        {"covariates": [[0.0], [NAN]]},
        {"covariate_names": [""]},
        {"onset_window_hours": (0.0, float("inf"))},
        {"onset_window_hours": (8.0, 0.0)},
        {"onset_window_hours": (0.0,)},
        {"onset_hours": [8.0, NAN]},
        {"onset_hours": [-0.5, NAN]},
    ],
)
def test_malformed_stays_are_refused(change) -> None:
    arguments = {
        "stay_ids": ["a", "b"],
        "group_ids": ["a", "b"],
        "onset_hours": [NAN, NAN],
        "death_hours": [NAN, NAN],
        "icu_exit_hours": [50.0, 50.0],
        "endpoint_observed": [True, True],
        "covariates": [[0.0], [1.0]],
        "covariate_names": ["x"],
        "onset_window_hours": (0.0, 8.0),
    }
    arguments.update(change)
    with pytest.raises(CloneCensorWeightError) as raised:
        trial_stays(**arguments)
    assert raised.value.code == "ccw_input_invalid"


@pytest.mark.parametrize("window", [(0.0, T0 + GRACE - 1.0), (1.0, 24.0)])
def test_starts_captured_short_of_the_grace_period_stop_the_estimate(window) -> None:
    sim = _simulate(500, 33)
    keep = (sim["onset"] >= window[0]) & (sim["onset"] < window[1])
    sim["onset"] = np.where(keep, sim["onset"], np.nan)
    with pytest.raises(CloneCensorWeightError) as raised:
        estimate_clone_censor_weight(_stays(sim, window=window), TIMING)
    assert raised.value.code == "ccw_onset_window_short"


def test_no_eligible_stay_and_no_start_are_named() -> None:
    with pytest.raises(CloneCensorWeightError) as raised:
        estimate_clone_censor_weight(
            _tiny(onset=[NAN], death=[T0 - 1.0], exit_=[50.0]), TIMING
        )
    assert raised.value.code == "ccw_no_eligible_stay"
    sim = _simulate(500, 31)
    sim["onset"] = np.full(500, np.nan)
    with pytest.raises(CloneCensorWeightError) as raised:
        estimate_clone_censor_weight(_stays(sim), TIMING)
    assert raised.value.code == "ccw_strategy_unobserved"


def _not_estimable(sim, **kwargs) -> Optional[str]:
    with pytest.raises(CloneCensorWeightError) as raised:
        estimate_clone_censor_weight(_stays(sim, **kwargs), TIMING)
    assert raised.value.code == "ccw_weight_model_not_estimable"
    assert raised.value.cause in WEIGHT_MODEL_NOT_ESTIMABLE_CAUSES
    return raised.value.cause


def test_a_weight_model_that_cannot_be_estimated_stops_with_its_cause() -> None:
    sim = _simulate(3000, 41)
    assert _not_estimable(sim, extra=sim["x1"] * 2.0) == "singular_design"
    never = np.zeros(3000)
    never[:400] = 1.0
    sim["onset"][:400] = np.nan
    assert _not_estimable(sim, extra=never) == "separation"


def test_hour_terms_follow_the_grace_period() -> None:
    assert hour_terms(np.arange(1), 1).shape == (1, 0)
    assert hour_terms(np.arange(3), 3).tolist() == [[0.0], [1.0], [2.0]]
    grace = MAX_GRACE_PERIOD_HOURS
    hours = np.arange(grace, dtype=float)
    basis = hour_terms(hours, grace)
    first, _, last = (
        fraction * (grace - 1)
        for fraction in TARGET_TRIAL_HOST_POLICY["hour_spline_knot_fractions"]
    )
    assert np.array_equal(basis[:, 0], hours)
    assert np.all(basis[hours <= first, 1] == 0.0)
    beyond = basis[hours >= last, 1]
    assert beyond.shape[0] >= 3
    assert np.allclose(np.diff(beyond, n=2), 0.0)
    assert np.all(np.diff(basis[:, 1]) >= 0.0)


# -- coverage ------------------------------------------------------------------


@pytest.mark.slow
def test_the_percentile_interval_covers_the_true_risk_difference() -> None:
    truth_initiate, truth_defer = _truth()
    truth = truth_initiate - truth_defer
    covered = []
    for seed in range(40):
        estimate = estimate_clone_censor_weight(
            _stays(_simulate(3000, 500 + seed)), TIMING
        )
        low, high = bootstrap_clone_censor_weight(
            estimate, resamples=100, seed=seed
        ).intervals.risk_difference
        covered.append(low <= truth <= high)
    assert np.mean(covered) >= 0.85
