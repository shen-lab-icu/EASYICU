"""A native eICU export binds hospital-first selection without exposing IDs."""

import hashlib
import json
import stat

import pandas as pd
import pytest

from easyicu.research_agent.acquisition.first_icu_stay import FLAG
from easyicu.webserver import raw_source_authority, source_identity_authority


@pytest.fixture
def export(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    pd.DataFrame({
        "patientunitstayid": [10, 11, 12, 13],
        "patienthealthsystemstayid": [1, 1, 2, 3],
        "unitvisitnumber": [2, 1, 1, 2],
        "uniquepid": ["same", "same", "same", "other"],
        "unittype": ["MICU"] * 4,
        "hospitaldischargeoffset": [100, 100, 100, 100],
        "hospitaldischargestatus": ["Alive"] * 4,
    }).to_parquet(raw / "patient.parquet", index=False)
    path = tmp_path / "export"
    path.mkdir()
    (path / "_manifest.json").write_text(json.dumps({
        "database": "eicu_demo", "data_path": str(raw), "files": [],
    }))
    monkeypatch.setattr(source_identity_authority.state_paths, "state_root", lambda: tmp_path / "state")
    monkeypatch.setattr(raw_source_authority, "_DEFAULT_CONFIG_PATH", tmp_path / "absent.env")
    return path


def test_native_demo_binding_is_private_digest_bound_and_does_not_change_patient_first(export):
    binding = source_identity_authority.resolve_study_first_icu_stay_per_hospitalization(
        export_path=export, database="eicu_demo",
    )
    assert binding is not None
    assert stat.S_IMODE(binding.coordinate_path.stat().st_mode) == 0o600
    assert hashlib.sha256(binding.coordinate_path.read_bytes()).hexdigest() == binding.coordinate_sha256
    assert pd.read_parquet(binding.coordinate_path).set_index("stay_id")[FLAG].to_dict() == {
        10: False, 11: True, 12: True, 13: False,
    }
    authority = dict(binding.authority_coordinates)
    assert authority["selection_rule"] == "minimum_declared_icu_unitvisitnumber_per_hospitalization"
    assert authority["first_icu_stays"] == 2
    assert authority["unknown_eligibility_stays"] == 1
    assert not authority["provider_visible_values"]
    assert "same" not in json.dumps(authority)
    assert source_identity_authority.resolve_study_first_icu_stay(export_path=export, database="eicu_demo") is None


def test_bound_raw_bytes_must_still_match(export):
    binding = raw_source_authority.resolve_raw_hospital_source_binding(export_path=export, database="eicu_demo")
    raw_path = export.parent / "raw" / "patient.parquet"
    changed = pd.read_parquet(raw_path)
    changed.loc[0, "unitvisitnumber"] = 1
    changed.to_parquet(raw_path, index=False)
    with pytest.raises(raw_source_authority.RawSourceAuthorityError):
        binding.materialize_first_icu_stay_per_hospitalization()


def test_missing_unit_type_does_not_default_to_icu(export):
    raw_path = export.parent / "raw" / "patient.parquet"
    pd.read_parquet(raw_path).drop(columns=["unittype"]).to_parquet(raw_path, index=False)
    assert source_identity_authority.resolve_study_first_icu_stay_per_hospitalization(
        export_path=export, database="eicu_demo",
    ) is None


def _study(export):
    return {
        "data_source": {"path": str(export), "database": "eicu_demo"},
        "cohort": {"preset": "adult_all", "age_min": 18, "exclude_readmissions": False,
                   "first_icu_stay_per_hospitalization": True},
    }


def test_readiness_uses_exact_coordinate_and_age_without_promoting_unknown(export, monkeypatch):
    from easyicu.webserver import data_package_execution_readiness as readiness

    ages = pd.DataFrame({"stay_id": [10, 11, 12, 13], "age": [50, 50, 17, 60]})
    monkeypatch.setattr(readiness, "_read_review_concept", lambda *_args: ages)
    result = readiness._cohort_eligibility_review(_study(export), source_path=str(export), registered_denominator=4)
    assert result["status"] == "ready" and result["count"] == 1
    assert result["excluded_reason_counts"] == {
        "non_first_icu_stay": 1, "hospitalization_prior_unit_coverage_incomplete": 1,
    }
    ages.loc[0, "stay_id"] = 999
    result = readiness._cohort_eligibility_review(_study(export), source_path=str(export), registered_denominator=4)
    assert result["status"] == "unavailable"
    assert result["reason_code"].endswith("coverage_incomplete")


def test_launch_requires_separate_hospital_authority_and_matching_materialized_receipt(export, tmp_path):
    from types import SimpleNamespace
    from easyicu.webserver.research_launch_scientific import _first_icu_stay_per_hospitalization_for_cohort
    from easyicu.webserver.agent_pipeline_runs import _require_first_icu_stay_materialized
    from easyicu.webserver.research_pipeline_run_errors import ResearchPipelineRunError

    binding = _first_icu_stay_per_hospitalization_for_cohort(_study(export))
    assert binding is not None
    path = tmp_path / "provenance.json"
    path.write_text(json.dumps({"first_icu_stay_restriction": {"coordinate_sha256": binding.coordinate_sha256}}))
    with pytest.raises(ResearchPipelineRunError):
        _require_first_icu_stay_materialized(SimpleNamespace(provenance_path=path), binding, per_hospitalization=True)
    path.write_text(json.dumps({"first_icu_stay_per_hospitalization_restriction": {
        "schema_version": "easyicu.first_icu_stay_per_hospitalization_restriction/1",
        "coordinate_sha256": binding.coordinate_sha256,
    }}))
    _require_first_icu_stay_materialized(SimpleNamespace(provenance_path=path), binding, per_hospitalization=True)


def test_planning_catalog_carries_the_same_icu_selection_policy_without_patient_rows(export, tmp_path):
    from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
    from easyicu.research_agent.research_context.builder import build_research_context
    from easyicu.webserver.agent_pipeline_runs import _metadata_only_planning_acquisition

    binding = source_identity_authority.resolve_study_first_icu_stay_per_hospitalization(
        export_path=export, database="eicu_demo",
    )
    acquisition = _metadata_only_planning_acquisition(
        database="eicu_demo", question="Synthetic hospital-first descriptive plan",
        output_dir=tmp_path / "planning", first_icu_stay_per_hospitalization=binding,
        llm=ScriptedMockLLMClient([json.dumps({
            "selected_concepts": ["age", "sex", "death"],
            "inclusion_exclusion": [], "rationale": "Synthetic contract check.",
        })]),
    )
    context = build_research_context(
        research_question="Synthetic hospital-first descriptive plan", cohort=acquisition.universe_path,
        database="eicu_demo", cohort_name="synthetic", primary_exposure="sex", target_outcome="death",
    )
    receipt = context.cohort.provenance["first_icu_stay_per_hospitalization_restriction"]
    assert context.cohort.n_stays == 0
    assert receipt["coordinate_sha256"] == binding.coordinate_sha256
    assert receipt["selection_rule"] == binding.authority_coordinates["selection_rule"]
    assert "MICU" in receipt["icu_unit_types"]
    assert receipt["non_icu_unit_types"] == ["SDU/Step down"]
    assert receipt["establishes_patient_independence"] is False
    assert "first_icu_stay_restriction" not in context.cohort.provenance


@pytest.mark.parametrize("cohort,reason", [
    ({"exclude_readmissions": True, "first_icu_stay_per_hospitalization": True},
     "conflicting_first_icu_stay_scope"),
    ({"first_icu_stay_per_hospitalization": "ambiguous"}, "invalid_cohort_boolean"),
])
def test_readiness_reports_an_invalid_scope_before_reading_any_source(monkeypatch, cohort, reason):
    from easyicu.webserver import data_package_execution_readiness as readiness

    def unexpected_read(*args, **kwargs):
        pytest.fail("An invalid cohort must stop before any source read")

    monkeypatch.setattr(readiness, "_read_review_concept", unexpected_read)
    result = readiness.build_data_package_execution_readiness(
        {"cohort": cohort}, source_path="/synthetic-unused", catalog_by_id={}, registered_denominator=4,
    )
    assert result["eligible_denominator"]["status"] == "unavailable"
    assert result["eligible_denominator"]["count"] is None
    assert result["eligible_denominator"]["reason_code"] == reason
    assert result["runtime_readiness"]["status"] == "blocked"
    assert result["runtime_readiness"]["required_findings"] == [reason]
