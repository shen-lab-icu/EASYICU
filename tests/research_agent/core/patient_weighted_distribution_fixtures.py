"""Synthetic declared design shared by marginal and contrast regressions."""

from easyicu.research_agent.schema import ExposureOutcomeDistributionSpec


ASSUMPTIONS = {
    "target": "conditional_expected_stay_proportion",
    "independence": "independent_patient_clusters_conditional_on_denominator",
    "weights": "fixed_observed_patient_stay_weights",
}


def _spec(**updates):
    payload = dict(
        exposure="group", exposure_levels=[0, 1], outcome="event",
        outcome_levels=[0, 1], outcome_positive_value=1,
        level_match_policy="exact_typed", denominator_policy="all_declared_rows",
        missing_outcome_policy="fail_closed", confidence_level=0.95,
        repeated_unit_interval_method="patient_weighted_hoeffding",
        patient_weighted_interval_assumptions=ASSUMPTIONS,
        dependence={"group_source": "patient", "group_derivation": "identity"},
    )
    payload.update(updates)
    return ExposureOutcomeDistributionSpec.model_validate(payload)
