"""What each home example asks of its database, shared by the starter and question-reader tests."""

# What each example asks of its database, as the capability owner reads it.
EXAMPLE_REQUIREMENTS = {
    "example-quick-lactate": ("hirid", "descriptive_epidemiology", {"concepts": ("lact",)}),
    "example-mortality-by-sex": ("mimic", "descriptive_epidemiology", {"outcome": "hospital_death", "concepts": ("sex",)}),
    "example-lactate-death": ("eicu", "association_study", {"outcome": "hospital_death", "concepts": ("lact", "age", "sex", "sofa")}),
    "example-sofa-trend": ("eicu", "ordinal_dose_response", {"outcome": "hospital_death", "concepts": ("sofa", "age", "sex")}),
    "example-lactate-dose-response": ("sic", "association_study", {"outcome": "hospital_death", "concepts": ("lact", "age", "sex")}),
    "example-landmark-lactate": ("miiv", "association_study", {"outcome": "mort_28d", "concepts": ("lact", "age", "sex", "map", "crea")}),
    "example-aki-survival": ("miiv", "survival", {"outcome": "mort_28d", "concepts": ("aki_stage_reference", "age", "sex"),
                                                  "censoring_unit": "hospitalization"}),
    "example-ventilation-rmst": ("sic", "survival", {"outcome": "mort_28d", "concepts": ("mech_vent", "age", "sex"),
                                                     "censoring_unit": "hospitalization"}),
    "example-ventilation-time-varying": ("mimic", "survival", {"outcome": "icu_death", "concepts": ("mech_vent", "age", "sex"),
                                                               "censoring_unit": "icu_stay"}),
    "example-creatinine-competing": ("sic", "survival", {"outcome": "icu_death", "concepts": ("crea", "age", "sex"),
                                                         "censoring_unit": "icu_stay"}),
    "example-predict-death": ("hirid", "prediction_model", {"outcome": "icu_death", "concepts": ("age", "sex", "map", "hr", "lact", "crea")}),
    "example-dynamic-prediction": ("aumc", "dynamic_prediction", {"outcome": "icu_death", "concepts": ("map", "hr", "lact", "crea", "age")}),
    "example-early-norepinephrine": ("miiv", "causal_inference", {"outcome": "mort_28d", "concepts": ("age", "sex", "lact", "map"),
                                                                  "treatment_concepts": ("norepi_rate",)}),
    "example-first-day-phenotypes": ("aumc", "trajectory_clustering", {"outcome": "icu_death",
                                                                       "concepts": ("map", "hr", "lact", "crea", "glu", "plt")}),
    "example-pressure-trajectories": ("eicu", "trajectory_clustering", {"outcome": "hospital_death", "concepts": ("map",)}),
    "example-vasopressor-use": ("aumc", "descriptive_epidemiology", {"concepts": ("vaso_ind",)}),
}
# A replication names its first database and the ones it repeats on; each
# must answer the same study exactly.
REPLICATION_EXAMPLES = {
    "example-replicate-lactate": tuple(
        (database, "cross_database_replication", {"outcome": "hospital_death", "concepts": ("lact", "age", "sex")})
        for database in ("miiv", "eicu", "sic")
    ),
}
