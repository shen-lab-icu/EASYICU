"""A synthetic long panel and cohort share the verified opaque stay identity."""

import hashlib

import pandas as pd

from easyicu.research_agent.acquisition.patient_grouping import PatientGroupingBinding
from easyicu.research_agent.cohort.materializer import materialize_to_parquet
from easyicu.research_agent.intake.materialized_trajectory import (
    load_verified_materialized_trajectory_authority,
    stage_materialized_trajectory_authority,
)
from easyicu.research_agent.intake.materialized_metadata import (
    stage_materialized_cohort_authority,
)
from tests.support.typed_export import typed_export


def test_grouping_retains_the_long_panel_with_exactly_the_cohort_stays(tmp_path):
    source = typed_export(
        tmp_path / "source",
        labs=pd.DataFrame(
            {
                "stay_id": [1, 1, 2, 3],
                "charttime": [1.0, 2.0, 1.0, 1.0],
                "age": [50, 50, 60, 70],
                "lact": [1.0, 2.0, 3.0, 4.0],
                "mech_vent": [False, False, True, False],
            }
        ),
        outcomes=pd.DataFrame({"stay_id": [1, 2, 3], "death": [False, True, False]}),
    )
    mapping = tmp_path / "mapping.parquet"
    pd.DataFrame({"stay_id": [1, 2, 3], "patient": [10, 10, 20]}).to_parquet(
        mapping, index=False
    )
    binding = PatientGroupingBinding(
        mapping_path=mapping,
        mapping_sha256=hashlib.sha256(mapping.read_bytes()).hexdigest(),
        mapping_stay_column="stay_id",
        mapping_patient_column="patient",
        authority_coordinates={},
    )
    paths = materialize_to_parquet(
        tmp_path / "prepared",
        stem="universe",
        database="miiv",
        data_path=source,
        static_concepts=("age",),
        feature_concepts=("lact",),
        outcome_concepts=("death",),
        emit_trajectory=True,
        trajectory_concepts=("lact",),
        trajectory_window=(0, 24),
        **binding.materializer_kwargs(),
    )
    cohort = pd.read_parquet(paths["parquet"])
    trajectory = pd.read_parquet(paths["trajectory"])
    assert (
        set(trajectory.stay_id)
        == set(cohort.patient_stay_id)
        == {"p10:s1", "p10:s2", "p20:s3"}
    )
    assert len(trajectory) == 4
    source_trajectory = load_verified_materialized_trajectory_authority(
        paths["trajectory"]
    )
    assert source_trajectory is not None
    staged_cohort = stage_materialized_cohort_authority(
        paths["parquet"],
        tmp_path / "run" / "cohort.parquet",
        producer_implementation_sha256="a" * 64,
    )
    staged = stage_materialized_trajectory_authority(
        paths["trajectory"],
        tmp_path / "run" / "cohort_trajectory.parquet",
        source_universe_path=paths["parquet"],
        target_universe_path=tmp_path / "run" / "cohort.parquet",
        expected_source_authority=source_trajectory.reference,
        expected_target_universe_authority=staged_cohort.reference,
        producer_implementation_sha256="a" * 64,
    )
    assert (
        staged.authority.trajectory_sha256
        == source_trajectory.authority.trajectory_sha256
    )
