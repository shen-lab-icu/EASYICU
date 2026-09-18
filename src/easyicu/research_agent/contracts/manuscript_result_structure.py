"""Neutral Results labels shared by planning-derived reports and claim grammar.

These exact structural labels confer no numerical or scientific authority.
Keep this vocabulary dependency-free; consumers retain their separate gates.
"""

PRIMARY_RESULT_HEADINGS_BY_FAMILY = {
    "descriptive_epidemiology": "Descriptive results",
    "prediction_model": "Model performance",
    "dynamic_prediction": "Model performance",
    "trajectory_clustering": "Cluster characteristics",
    "survival": "Survival results",
    "association_study": "Primary association",
    "ordinal_dose_response": "Primary association",
}
DEFAULT_PRIMARY_RESULT_HEADING = "Primary results"
COHORT_RESULT_HEADING = "Cohort characteristics"
RESULT_HEADINGS_BY_ROLE = {
    "secondary": "Secondary analyses",
    "sensitivity": "Sensitivity and subgroup analyses",
}
PRIMARY_RESULT_HEADINGS = tuple(dict.fromkeys((
    *PRIMARY_RESULT_HEADINGS_BY_FAMILY.values(), DEFAULT_PRIMARY_RESULT_HEADING,
)))
PLAN_RESULT_HEADINGS = frozenset((
    *PRIMARY_RESULT_HEADINGS, COHORT_RESULT_HEADING, *RESULT_HEADINGS_BY_ROLE.values(),
))
