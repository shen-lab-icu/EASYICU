"""Preparation counts need the same sealed row universe as the displayed run."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from easyicu.webserver import research_population_summary as owner


def _verified():
    parameters = {"first_icu_stay_per_hospitalization_restriction": {
            "schema_version": "easyicu.first_icu_stay_per_hospitalization_restriction/1",
            "stays_before": 10, "stays_after": 7,
            "coordinate_sha256": "a" * 64,
            "excluded_reason_counts": {"non_first_icu_stay": 2,
                                       "hospitalization_prior_unit_coverage_incomplete": 1},
            "authority_coordinates": {"coordinate_sha256": "a" * 64,
                                      "establishes_patient_independence": False,
                                      "private_path": "/private/source"},
        }}
    return SimpleNamespace(
        authority=SimpleNamespace(cohort_rows=7, producer_parameters=parameters),
        reference=SimpleNamespace(sha256="b" * 64),
    )


def test_counts_project_only_the_verified_bounded_phase(monkeypatch, tmp_path):
    monkeypatch.setattr(owner, "load_verified_materialized_cohort_authority", lambda path: _verified())
    result = owner.prepared_population_summary(SimpleNamespace(universe_path=tmp_path / "cohort.parquet"))
    assert result["stays_before"] == 10 and result["stays_after"] == 7
    assert sum(result["excluded_reason_counts"].values()) == 3
    assert result["establishes_patient_independence"] is False
    assert "/private/" not in str(result)


@pytest.mark.parametrize("failure", ["missing", "unreadable", "count", "boolean", "rows", "coordinate", "patient", "unknown_reason", "schema", "negative", "overflow", "coordinate_format", "authority"])
def test_missing_or_inconsistent_proof_cannot_issue_population_counts(monkeypatch, tmp_path, failure):
    verified = deepcopy(_verified())
    restriction = verified.authority.producer_parameters["first_icu_stay_per_hospitalization_restriction"]
    if failure == "missing":
        verified = None
    elif failure == "schema":
        restriction["schema_version"] = "unknown"
    elif failure == "negative":
        restriction["stays_before"] = -1
        restriction["excluded_reason_counts"]["non_first_icu_stay"] = -9
    elif failure == "overflow":
        restriction["stays_before"] += 2**53
        restriction["excluded_reason_counts"]["non_first_icu_stay"] += 2**53
    elif failure == "coordinate_format":
        restriction["coordinate_sha256"] = "bad"
        restriction["authority_coordinates"]["coordinate_sha256"] = "bad"
    elif failure == "authority":
        restriction["authority_coordinates"] = None
    elif failure == "count":
        restriction["stays_before"] = 11
    elif failure == "boolean":
        restriction["excluded_reason_counts"]["hospitalization_prior_unit_coverage_incomplete"] = True
    elif failure == "rows":
        verified.authority.cohort_rows = 8
    elif failure == "coordinate":
        restriction["coordinate_sha256"] = "c" * 64
    elif failure == "patient":
        restriction["authority_coordinates"]["establishes_patient_independence"] = True
    elif failure == "unknown_reason":
        restriction["excluded_reason_counts"]["/private/source"] = 0

    def load(path):
        if failure == "unreadable":
            raise OSError("unreadable authority")
        return verified

    monkeypatch.setattr(owner, "load_verified_materialized_cohort_authority", load)
    assert owner.prepared_population_summary(SimpleNamespace(universe_path=tmp_path / "cohort.parquet")) is None


def test_actual_materialization_counts_are_digest_bound_to_the_prepared_rows(tmp_path):
    import hashlib
    import json
    import pandas as pd
    from easyicu.research_agent.acquisition.first_icu_stay import derive_first_icu_stay_per_hospitalization
    from easyicu.research_agent.cohort.materializer import materialize_to_parquet
    from easyicu.webserver.agent_pipeline_runs import _write_projection
    from tests.support.typed_export import typed_export

    source = typed_export(tmp_path / "export")
    derived = derive_first_icu_stay_per_hospitalization(pd.DataFrame({
        "patientunitstayid": [1, 2], "patienthealthsystemstayid": [1, 1],
        "unitvisitnumber": [2, 1], "unittype": ["MICU", "MICU"],
    }))
    coordinate = tmp_path / "coordinate.parquet"
    derived.frame.to_parquet(coordinate, index=False)
    digest = hashlib.sha256(coordinate.read_bytes()).hexdigest()
    paths = materialize_to_parquet(
        tmp_path / "materialized", data_path=source, database="miiv",
        static_concepts=("age",), feature_concepts=("lact",), outcome_concepts=("death",),
        first_icu_stay_per_hospitalization_path=coordinate,
        first_icu_stay_per_hospitalization_sha256=digest,
        first_icu_stay_per_hospitalization_authority_coordinates={
            **derived.receipt, "coordinate_sha256": digest,
        },
    )
    acquisition = SimpleNamespace(universe_path=paths["parquet"])
    summary = owner.prepared_population_summary(acquisition)
    assert (summary["stays_before"], summary["stays_after"]) == (2, 1)
    assert summary["excluded_reason_counts"] == {"non_first_icu_stay": 1}
    assert summary["coordinate_sha256"] == digest
    assert str(tmp_path) not in str(summary)
    wrapper = tmp_path / "run_synthetic_projection"
    _write_projection(
        wrapper_dir=wrapper,
        study={"id": "synthetic", "question": "Describe the synthetic population",
               "data_source": {"database": "miiv", "label": "synthetic"}},
        provider={}, acquisition=acquisition, run_dir=None,
    )
    for filename in ("run_context.json", "cohort_summary.json"):
        projected = json.loads((wrapper / filename).read_text())
        assert projected["population_preparation"] == summary
    paths["parquet"].write_bytes(b"changed synthetic cohort")
    assert owner.prepared_population_summary(acquisition) is None
