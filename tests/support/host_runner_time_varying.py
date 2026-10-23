"""Synthetic counting-process inputs for local runner parity."""

from __future__ import annotations
import hashlib
import json
import numpy as np
import pandas as pd
import pytest
from easyicu.research_agent.acquisition.foundation import AcquisitionResult
from easyicu.research_agent.acquisition.hospital_mortality_followup import (
    HospitalMortalityFollowup,
)
from easyicu.research_agent.acquisition.patient_grouping import PatientGroupingBinding
from easyicu.research_agent.acquisition.time_varying_materialization import (
    materialize_time_varying_acquisition,
)
from easyicu.research_agent.canonical_json import sha256_file
from easyicu.research_agent.contracts.dependence import PlannedDependenceRequirement
from easyicu.research_agent.contracts.time_varying_exposure import (
    TimeVaryingExposureSpecification,
)
from easyicu.research_agent.planning.sensitivity_authority import (
    PrespecifiedSensitivitySpec,
)
from easyicu.webserver.scientific_runtime_projection import (
    compile_web_scientific_runtime_projection,
)


def time_varying_spec():
    return TimeVaryingExposureSpecification.model_validate(
        {
            "exposure_concept": "lact",
            "exposure_summary": "running_max_of_direct_measurements",
            "exposure_window_hours": 24,
            "followup": "hospital_death_or_discharge_from_icu_admission",
            "missingness_policy": "observed_state_indicator",
            "baseline_columns": ["age", "sex"],
            "baseline_categorical_encodings": {
                "sex": {
                    "kind": "binary_indicator",
                    "output_column": "sex_binary",
                    "positive_level": "Male",
                    "negative_level": "Female",
                    "unknown_or_missing_policy": "reject",
                }
            },
            "interpretation": "descriptive_time_updated_association_not_causal",
        }
    )


def time_varyingsource(tmp_path):
    rng = np.random.default_rng(1279)
    n = 300
    stay = np.arange(100001, 100001 + n)
    baseline = pd.DataFrame(
        {
            "stay_id": stay,
            "age": rng.uniform(20, 80, n),
            "sex": np.where(rng.random(n) < 0.5, "Female", "Male"),
        }
    )
    cohort_path = tmp_path / "source_cohort.parquet"
    baseline.to_parquet(cohort_path, index=False)
    provenance_path = tmp_path / "source_provenance.json"
    provenance_path.write_text(
        json.dumps(
            {
                "schema_version": "easyicu.cohort_materializer/1",
                "cohort_window_hours": [0.0, 24.0],
                "feature_concepts": ["lact"],
                "outcome_concepts": ["death"],
                "static_concepts": ["age", "sex"],
                "n_stays_after_inclusion_exclusion": len(baseline),
                "columns": list(baseline),
                "cohort_sha256": hashlib.sha256(
                    pd.util.hash_pandas_object(
                        baseline.reset_index(drop=True), index=False
                    ).values.tobytes()
                ).hexdigest(),
                "cohort_file_sha256": sha256_file(cohort_path),
                "cohort_file_size": cohort_path.stat().st_size,
                "producer_parameters": {},
            }
        )
    )
    measured = stay[np.arange(n) % 3 != 0]
    trajectory = pd.DataFrame(
        {
            "stay_id": measured,
            "charttime": 2.0,
            "concept": "lact",
            "value_num": rng.uniform(0.7, 6, len(measured)),
            "value_str": None,
            "evidence_state": "direct_observed",
            "owner_observed": True,
            "owner_available": True,
        }
    )
    trajectory_path = tmp_path / "source_trajectory.parquet"
    trajectory.to_parquet(trajectory_path, index=False)
    mapping_path = tmp_path / "private_mapping.parquet"
    pd.DataFrame(
        {"stay_id": stay, "patient_key": np.arange(n) // 2 + 91000}
    ).to_parquet(mapping_path, index=False)
    mapping_sha = sha256_file(mapping_path)
    grouping = PatientGroupingBinding(
        mapping_path=mapping_path,
        mapping_sha256=mapping_sha,
        mapping_stay_column="stay_id",
        mapping_patient_column="patient_key",
        authority_coordinates={
            "schema_version": "easyicu.patient_grouping_runtime_authority/1",
            "authority_ref": "test/private-map",
            "mapping_sha256": mapping_sha,
            "grouping_derivation": "prefix_before_:s",
            "provider_visible_values": False,
        },
    )
    duration = rng.uniform(3, 100, n)
    event = (rng.random(n) < 0.45).astype("int8")
    duration[0], event[0] = (0.5, 1)
    followup = HospitalMortalityFollowup(
        frame=pd.DataFrame(
            {
                "stay_id": stay,
                "hospital_death": event,
                "death_time_hours": np.where(event, duration, np.nan),
                "hospital_followup_time_hours": duration,
            }
        ),
        exclusions=pd.DataFrame(columns=["stay_id", "reason_code"]),
        receipt={
            "schema_version": "easyicu.mimic_iv_hospital_mortality_followup/1",
            "database": "synthetic",
            "analysis_unit": "icu_stay",
            "time_origin": "icu_admission",
            "time_unit": "hours",
            "event": {
                "column": "hospital_death",
                "definition": "synthetic binary event",
                "event_time_column": "death_time_hours",
                "event_time_source": "synthetic",
            },
            "censoring": {
                "followup_time_column": "hospital_followup_time_hours",
                "rule": "event else administrative censoring",
                "source": "synthetic",
            },
            "input_stays": n,
            "valid_stays": n,
            "excluded_stays": 0,
            "event_stays": int(event.sum()),
            "censored_stays": int(n - event.sum()),
            "zero_time_event_stays": 0,
            "zero_time_censored_stays": 0,
            "exclusion_counts": {},
            "privacy": {
                "identifier_values_returned": False,
                "raw_rows_returned": False,
                "source_paths_returned": False,
            },
        },
    )
    acquisition = AcquisitionResult(
        universe_path=cohort_path,
        provenance_path=provenance_path,
        trajectory_path=trajectory_path,
        trajectory_provenance_path=provenance_path,
        selection=None,
        coverage=None,
        materialized_concepts=["lact", "age", "sex", "death"],
    )
    return (acquisition, followup, grouping)


def time_varying_materialize(time_varyingsource):
    acquisition, followup, grouping = time_varyingsource
    return materialize_time_varying_acquisition(
        acquisition,
        specification=time_varying_spec(),
        hospital_followup=followup,
        patient_grouping=grouping,
        raw_source_receipt={"authority_ref": "synthetic/raw"},
        exposure_column="lact",
    )


def time_varying_projection(
    path,
    *,
    cohort=None,
    literature_citation_keys=(),
    direct_comparator_literature_keys=(),
):
    row = PrespecifiedSensitivitySpec.model_validate(
        {
            "spec_id": "time_varying_exposure",
            "axis": "timing",
            "strategy": "time_varying",
            "execution_variables": ["lact"],
            "time_varying_execution": time_varying_spec().model_dump(mode="json"),
        }
    )
    return compile_web_scientific_runtime_projection(
        study={"covariate_selection": "exact", "cohort": cohort or {}},
        sensitivity_specs=[row],
        primary_exposure="lact",
        primary_exposure_source="lact",
        target_outcome="death",
        declared_covariates=["age", "sex"],
        covariate_operationalizations={},
        target_is_event_status=True,
        universe_path=path,
        scientific_configuration_sha256="a" * 64,
        literature_citation_keys=literature_citation_keys,
        direct_comparator_literature_keys=direct_comparator_literature_keys,
        dependence=PlannedDependenceRequirement(
            group_source="patient_stay_id",
            group_derivation="prefix_before_delimiter",
            delimiter=":s",
        ),
    )
