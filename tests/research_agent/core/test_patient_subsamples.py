import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score

from easyicu.research_agent.contracts.dependence import PlannedDependenceRequirement
from easyicu.research_agent.methods.patient_subsamples import (
    patient_units,
    patient_subsample_positions,
)


def test_repeated_stays_are_never_split_between_a_subsample_and_its_complement():
    requirement = PlannedDependenceRequirement(
        group_source="patient_stay_id",
        group_derivation="prefix_before_delimiter",
        delimiter=":s",
    )
    ids = ["p1:s1", "p1:s2", "p2:s3", "p3:s4", "p3:s5", "p3:s6"]
    groups, units = patient_units(ids, requirement)
    for seed in range(20):
        positions = patient_subsample_positions(
            groups, units, sample_n=2, rng=np.random.default_rng(seed)
        )
        picked = set(positions)
        assert len(set(groups[positions])) == 2
        assert not set(groups[positions]) & {
            group for i, group in enumerate(groups) if i not in picked
        }


def test_single_stay_patient_resampling_preserves_previous_stability_scores():
    rng = np.random.default_rng(732)
    panel = rng.normal(size=(60, 4))
    reference = GaussianMixture(
        n_components=3, covariance_type="diag", random_state=1, n_init=10
    ).fit_predict(panel)
    requirement = PlannedDependenceRequirement(
        group_source="patient", group_derivation="identity"
    )
    groups, units = patient_units(np.arange(len(panel)), requirement)
    for seed in range(5):
        old = np.sort(
            np.random.default_rng(seed).choice(len(panel), size=40, replace=False)
        )
        actual = patient_subsample_positions(
            groups, units, sample_n=40, rng=np.random.default_rng(seed)
        )
        np.testing.assert_array_equal(actual, old)
        old_score = adjusted_rand_score(
            reference[old],
            GaussianMixture(
                n_components=3, covariance_type="diag", random_state=seed, n_init=10
            ).fit_predict(panel[old]),
        )
        actual_score = adjusted_rand_score(
            reference[actual],
            GaussianMixture(
                n_components=3, covariance_type="diag", random_state=seed, n_init=10
            ).fit_predict(panel[actual]),
        )
        assert actual_score == old_score


def test_correlated_stays_reduce_the_number_of_independent_sampling_units():
    requirement = PlannedDependenceRequirement(
        group_source="patient", group_derivation="identity"
    )
    groups, units = patient_units(np.repeat(np.arange(20), 3), requirement)
    for seed in range(5):
        actual = patient_subsample_positions(
            groups, units, sample_n=10, rng=np.random.default_rng(seed)
        )
        # Half the patients implies half the stays, with all three copies
        # selected together; the previous stay sampler splits patients.
        assert len(actual) == 30
        assert len(set(groups[actual])) == 10
        old = np.random.default_rng(seed).choice(len(groups), size=30, replace=False)
        assert len(set(groups[old])) > 10
