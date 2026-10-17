"""Synthetic boundary groups retain uncertainty at every contrast consumer."""

import json
from pathlib import Path

import pandas as pd
import pytest

from easyicu.research_agent.authority.descriptive_scientific_claims import (
    derive_descriptive_claim_payloads,
)
from easyicu.research_agent.execution.runners.exposure_outcome_distribution_executor import (
    _distribution_rows,
    _proportion_interval,
    _verify_product,
)
from easyicu.research_agent.execution.runners.exposure_outcome_distribution_render import _validate
from easyicu.research_agent.schema import ExposureOutcomeDistributionSpec

from .patient_weighted_distribution_fixtures import _spec


@pytest.mark.parametrize("boundary", [0, 1])
@pytest.mark.parametrize("boundary_level", [0, 1])
@pytest.mark.parametrize("clustered", [False, True])
def test_one_boundary_group_refuses_wald_even_when_the_other_group_has_variance(
    boundary, boundary_level, clustered,
):
    frame = pd.DataFrame({
        "patient": [f"mock-{i}" for i in range(12)],
        "group": [0] * 6 + [1] * 6,
        "event": ([boundary] * 6 + [0, 1] * 3 if boundary_level == 0
                  else [0, 1] * 3 + [boundary] * 6),
    })
    spec = _spec(risk_difference_contrast={
        "reference_exposure_level": 0, "comparison_exposure_level": 1,
    })
    if not clustered:
        spec = _spec(
            dependence=None, repeated_unit_interval_method="patient_cluster_robust_wald",
            patient_weighted_interval_assumptions=None,
            risk_difference_contrast=spec.risk_difference_contrast,
        )
    with pytest.raises(RuntimeError, match="zero-width uncertainty"):
        _distribution_rows(frame, spec=spec)


@pytest.mark.parametrize("clustered", [False, True])
def test_two_rows_in_one_group_do_not_gain_wald_support_from_a_larger_other_group(clustered):
    frame = pd.DataFrame({"patient": [f"mock-{i}" for i in range(8)],
                          "group": [0, 0] + [1] * 6, "event": [0, 1] * 4})
    updates = {} if clustered else {
        "dependence": None, "repeated_unit_interval_method": "patient_cluster_robust_wald",
        "patient_weighted_interval_assumptions": None,
    }
    spec = _spec(**updates, risk_difference_contrast={
        "reference_exposure_level": 0, "comparison_exposure_level": 1,
    })
    with pytest.raises(RuntimeError, match="more than two analysed rows in each group"):
        _distribution_rows(frame, spec=spec)


@pytest.fixture
def legacy_product():
    path = Path(__file__).parents[1] / "fixtures/boundary_wald_legacy_product.json"
    return json.loads(path.read_text())


def test_product_verification_rejects_a_previously_emitted_boundary_contrast(legacy_product):
    with pytest.raises(RuntimeError, match="zero-width uncertainty"):
        _verify_product(
            legacy_product["table"],
            spec=ExposureOutcomeDistributionSpec.model_validate(legacy_product["spec"]),
        )


def test_rendering_rejects_a_previously_emitted_boundary_contrast(legacy_product):
    with pytest.raises(ValueError, match="zero-width uncertainty"):
        _validate(pd.DataFrame(legacy_product["table"]))


def test_claims_reject_a_previously_emitted_boundary_contrast(legacy_product):
    with pytest.raises(ValueError, match="zero-width uncertainty"):
        derive_descriptive_claim_payloads(legacy_product["summary"])


@pytest.mark.parametrize("events", [[0], [1], [0, 0], [0, 1], [1, 1]])
@pytest.mark.parametrize("method", ["patient_weighted_hoeffding", "patient_cluster_robust_wald"])
def test_tiny_stratum_policy_is_consistent_at_boundary_and_interior_proportions(events, method):
    frame = pd.DataFrame({"patient": [f"mock-{i}" for i in range(len(events))], "event": events})
    spec = _spec(repeated_unit_interval_method=method,
                 patient_weighted_interval_assumptions=(
                     _spec().patient_weighted_interval_assumptions
                     if method == "patient_weighted_hoeffding" else None
                 ))
    kwargs = {"spec": spec, "denominator_mask": frame.event.notna(),
              "numerator_mask": frame.event.eq(1)}
    if len(events) == 2 and method == "patient_weighted_hoeffding":
        interval = _proportion_interval(frame, **kwargs)
        low, high = interval["ci_low_pct"], interval["ci_high_pct"]
        assert high - low >= 96.0
        assert low <= 100 * sum(events) / len(events) <= high
    else:
        with pytest.raises(RuntimeError):
            _proportion_interval(frame, **kwargs)
