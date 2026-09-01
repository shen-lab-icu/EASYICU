from __future__ import annotations

from pathlib import Path


def test_experiment_spec_roundtrip_yaml(ra, tmp_path: Path):
    spec = ra.ExperimentSpec(
        question="Is admission SOFA associated with ICU mortality?",
        cohort=ra.CohortInputSpec(
            cohort="/tmp/cohort.parquet",
            cohort_name="demo",
            database="miiv",
            target_outcome="death",
            user_preferences={"inferred_analysis_family": "association_study"},
        ),
    )
    path = tmp_path / "experiment.yaml"
    ra.dump_experiment_spec(spec, path)
    loaded = ra.load_experiment_spec(path)
    assert loaded.question == spec.question
    assert loaded.cohort.database == "miiv"
    assert loaded.cohort.user_preferences["inferred_analysis_family"] == "association_study"


def test_experiment_spec_keeps_standard_executor_timeout_independent(ra):
    spec = ra.ExperimentSpec(
        question="Assess a frozen resampling design.",
        cohort=ra.CohortInputSpec(cohort="/tmp/cohort.parquet"),
        runtime=ra.RuntimeSpec(
            timeout_seconds=23.0,
            standard_executor_timeout_seconds=1_234.0,
        ),
    )

    kwargs = spec.pipeline_kwargs()
    assert kwargs["timeout_seconds"] == 23.0
    assert kwargs["standard_executor_timeout_seconds"] == 1_234.0
