"""Cross-field scientific ceilings shared by configuration and execution."""


class AnalysisDesignConflict(ValueError):
    code = "analysis_design_counts_only_family_conflict"


def validate_analysis_family_ceiling(
    *, analysis_family: str | None, variance_estimator: str
) -> None:
    """Counts-only is not a way to bypass a prediction/association contract.

    The caller canonicalizes family names. An absent family is a legacy
    coordinate, not permission to infer one; downstream step guards still run.
    """
    if (
        variance_estimator == "none_counts_only"
        and analysis_family
        and analysis_family != "descriptive_epidemiology"
    ):
        raise AnalysisDesignConflict(
            "A counts-only ceiling is compatible only with descriptive epidemiology; the agent must revise the design without changing the research question."
        )
