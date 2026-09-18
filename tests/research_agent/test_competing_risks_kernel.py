"""Kernel tests: competing-risks CIF via lifelines Aalen-Johansen.

Validity anchors:

* hand-computed CIF: durations [1, 2, 3], events [1, 2, 0], interest 1 ->
  CIF_1(t) = 1/3 for t >= 1 (S(0)=1, risk set 3, one type-1 event at t=1).
* degeneracy: with no competing event, CIF_1 == 1 - KM (lifelines
  KaplanMeierFitter cross-check).
* wiring cross-check: curve matches a direct lifelines AalenJohansenFitter
  fit on competing data.
"""

from __future__ import annotations

import numpy as np
import pytest

from easyicu.research_agent.methods.competing_risks import (
    cif_difference,
    estimate_cif,
)

D_TINY = [1.0, 2.0, 3.0]
E_TINY = [1, 2, 0]

D_A = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
E_A = [1, 0, 1, 2, 1, 0, 2, 1, 0, 1]
D_B = [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5]
E_B = [0, 2, 0, 2, 0, 1, 0, 2, 0, 0]
D_GROUPS = D_A + D_B
E_GROUPS = E_A + E_B
G_GROUPS = ["A"] * 10 + ["B"] * 10


def test_cif_hand_computed_tiny_case() -> None:
    res = estimate_cif(D_TINY, E_TINY, event_of_interest=1)
    assert res.times[0] == pytest.approx(0.0)
    assert res.cif[0] == pytest.approx(0.0)
    got = dict(zip(res.times, res.cif))
    assert got[1.0] == pytest.approx(1.0 / 3.0)
    assert got[2.0] == pytest.approx(1.0 / 3.0)  # t=2 event is competing
    assert got[3.0] == pytest.approx(1.0 / 3.0)
    assert res.n_interest == 1 and res.n_competing == 1 and res.n_censored == 1
    assert res.n == 3
    assert all(0.0 <= v <= 1.0 for v in res.cif)
    assert res.claim_ceiling == "analysis_only"


def test_cif_matches_lifelines_direct_fit() -> None:
    from lifelines import AalenJohansenFitter

    res = estimate_cif(D_GROUPS, E_GROUPS, event_of_interest=1, random_state=0)
    ref = AalenJohansenFitter(seed=0).fit(D_GROUPS, E_GROUPS, event_of_interest=1)
    ref_times = np.asarray(ref.cumulative_density_.index.to_numpy(dtype=float))
    ref_vals = np.asarray(ref.cumulative_density_.iloc[:, 0].to_numpy(dtype=float))
    np.testing.assert_allclose(list(res.times), ref_times, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(list(res.cif), ref_vals, rtol=1e-12, atol=1e-12)


def test_cif_degenerates_to_one_minus_km_without_competition() -> None:
    from lifelines import KaplanMeierFitter

    durations = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    events = [1, 0, 1, 1, 0, 1, 0, 1]
    res = estimate_cif(durations, events, event_of_interest=1)
    kmf = KaplanMeierFitter().fit(durations, [1 if e == 1 else 0 for e in events])
    for t, v in zip(res.times, res.cif):
        km = float(kmf.survival_function_at_times(t).iloc[0])
        assert v == pytest.approx(1.0 - km, abs=1e-9)


def test_cif_deterministic_including_ties() -> None:
    durations = [1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 5.0]
    events = [1, 2, 1, 0, 2, 1, 0, 2]
    first = estimate_cif(durations, events, event_of_interest=2, random_state=0)
    second = estimate_cif(durations, events, event_of_interest=2, random_state=0)
    assert first.to_json() == second.to_json()


def test_timeline_read_matches_step_curve() -> None:
    res = estimate_cif(D_TINY, E_TINY, event_of_interest=1, timeline=[0.5, 1.5, 2.5])
    assert tuple(res.times) == (0.5, 1.5, 2.5)
    assert list(res.cif) == pytest.approx([0.0, 1.0 / 3.0, 1.0 / 3.0])


def test_group_difference_descriptive_with_ci() -> None:
    res = cif_difference(
        D_GROUPS, E_GROUPS, G_GROUPS,
        event_of_interest=1, time=8.0, n_bootstrap=200, random_state=0,
    )
    assert res.diff == pytest.approx(res.cif_a - res.cif_b)
    assert res.diff > 0.0  # group A concentrates early cause-1 events
    assert res.ci_low <= res.diff <= res.ci_high
    assert res.se >= 0.0
    assert res.n_successful >= 100
    assert res.group_a == "A" and res.group_b == "B"
    assert res.gray_test is False
    assert res.hypothesis_test is None
    payload = res.to_json()
    assert payload["gray_test"] is False
    assert payload["hypothesis_test"] is None
    assert payload["claim_ceiling"] == "analysis_only"
    again = cif_difference(
        D_GROUPS, E_GROUPS, G_GROUPS,
        event_of_interest=1, time=8.0, n_bootstrap=200, random_state=0,
    )
    assert res.to_json() == again.to_json()


@pytest.mark.parametrize(
    "durations,events,code",
    [
        ([1.0, 2.0], [1], 1),  # length mismatch
        ([-1.0, 2.0], [1, 0], 1),  # negative duration
        ([1.0, np.nan], [1, 0], 1),  # NaN duration
        ([1.0, 2.0], [1, 1.5], 1),  # non-integral event code
        ([1.0, 2.0], [1, -1], 1),  # negative event code
        ([1.0, 2.0], [2, 0], 1),  # unobserved event of interest
        ([1.0, 2.0], [1, 0], 0),  # zero is censoring, not an event
    ],
)
def test_fail_closed_illegal_cif_inputs(
    durations: object, events: object, code: object
) -> None:
    with pytest.raises(ValueError):
        estimate_cif(durations, events, event_of_interest=code)  # type: ignore[arg-type]


def test_fail_closed_group_comparison() -> None:
    with pytest.raises(ValueError):  # three groups
        cif_difference(
            D_GROUPS, E_GROUPS, ["A"] * 7 + ["B"] * 7 + ["C"] * 6,
            event_of_interest=1, time=8.0, n_bootstrap=50,
        )
    with pytest.raises(ValueError):  # one group
        cif_difference(
            D_A, E_A, ["A"] * 10,
            event_of_interest=1, time=8.0, n_bootstrap=50,
        )
    with pytest.raises(ValueError):  # negative time
        cif_difference(
            D_GROUPS, E_GROUPS, G_GROUPS,
            event_of_interest=1, time=-1.0, n_bootstrap=50,
        )
    with pytest.raises(ValueError):  # group without the event of interest
        cif_difference(
            D_A + D_B, E_A + [0] * 10, G_GROUPS,
            event_of_interest=1, time=8.0, n_bootstrap=50,
        )
    with pytest.raises(ValueError):  # decreasing timeline
        estimate_cif(D_TINY, E_TINY, event_of_interest=1, timeline=[2.0, 1.0])
