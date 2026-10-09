"""Cross-field scientific ceilings shared by configuration and execution."""


class AnalysisDesignConflict(ValueError):
    code = "analysis_design_counts_only_family_conflict"


class BootstrapFamilyConflict(AnalysisDesignConflict):
    code = "analysis_design_bootstrap_family_conflict"


def validate_analysis_family_ceiling(
    *, analysis_family: str | None, variance_estimator: str
) -> None:
    """Counts-only is not a way to bypass a prediction/association contract.

    The caller canonicalizes family names. An absent family is a legacy
    coordinate, not permission to infer one; downstream step guards still run.
    A bootstrap is the interval only the causal inference suite computes, so
    only a declared causal family states it; no legacy design names one.
    """
    if (
        variance_estimator == "none_counts_only"
        and analysis_family
        and analysis_family != "descriptive_epidemiology"
    ):
        raise AnalysisDesignConflict(
            "A counts-only ceiling is compatible only with descriptive epidemiology; the agent must revise the design without changing the research question."
        )
    if variance_estimator == "bootstrap" and analysis_family != "causal_inference":
        raise BootstrapFamilyConflict(
            "A bootstrap is the interval of the causal inference suite only; the agent must revise the design without changing the research question."
        )
