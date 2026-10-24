"""Shared declared missing-value policies for synthetic descriptive plans."""

EXCLUDE_MISSING = {
    "denominator_policy": "observed_outcome_rows",
    "missing_exposure_policy": "exclude_from_denominator",
    "missing_outcome_policy": "exclude_from_denominator",
}
STOP_MISSING = {
    "denominator_policy": "all_declared_rows",
    "missing_exposure_policy": "fail_closed",
    "missing_outcome_policy": "fail_closed",
}


def missing_policies_for(request):
    """Keep the request's declaration, or explicitly stop on missing data."""
    return (request.distribution_missing_policies.model_dump()
            if request.distribution_missing_policies is not None else dict(STOP_MISSING))
