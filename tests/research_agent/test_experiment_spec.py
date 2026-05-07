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
