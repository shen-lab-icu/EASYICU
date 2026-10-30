import pandas as pd
import pytest

from easyicu.concept.sic_landmark import sic_case_landmark_fields
from easyicu.research_agent.authority.care_unit_landmark import case_duration_landmark_risk_set


def test_sic_case_fields_are_nullable_and_bind_same_case_event_horizon():
    cases = pd.DataFrame({
        "CaseID": range(1, 9),
        "ICUOffset": [3600] * 8,
        "TimeOfStay": [3600 + 48 * 3600, 3600 + 24 * 3600, 3600 + 48 * 3600,
                       3600 + 48 * 3600, 3600 + 48 * 3600, 3600 + 48 * 3600,
                       None, 3600 + 48 * 3600],
        "OffsetOfDeath": [None, None, 3600 + 12 * 3600, None, 3600 + 72 * 3600,
                          3600 + 12 * 3600, None, float("inf")],
        "HospitalDischargeType": [2026, 2026, 2028, 2028, 2028, 2026, 2026, 2026],
    })
    result = sic_case_landmark_fields(cases)
    assert result["care_unit_to_case_closure_days"].iloc[:3].tolist() == [2, 1, 2]
    assert result["death_during_case"].iloc[:5].tolist() == [False, False, True, pd.NA, False]
    assert result["death_time_during_case_hours"].iloc[2] == 12
    assert pd.isna(result["death_time_during_case_hours"].iloc[4])
    assert result["death_during_case"].iloc[5:].isna().all()
    risk = case_duration_landmark_risk_set(
        duration_hours=result["care_unit_to_case_closure_days"] * 24,
        death_status=result["death_during_case"], death_time_hours=result["death_time_during_case_hours"],
        landmark_hours=24,
    )
    assert risk.eligible.tolist() == [True, False, False, False, True, False, False, False]
    assert risk.unidentifiable.tolist() == [False, False, False, True, False, True, True, True]


def test_sic_case_fields_refuse_missing_or_ambiguous_case_identity():
    with pytest.raises(ValueError, match="source fields missing"):
        sic_case_landmark_fields(pd.DataFrame({"CaseID": [1]}))
    cases = pd.DataFrame({"CaseID": [1, 1], "ICUOffset": [0, 0], "TimeOfStay": [1, 1],
                          "OffsetOfDeath": [None, None], "HospitalDischargeType": [2026, 2026]})
    with pytest.raises(ValueError, match="unique non-null CaseID"):
        sic_case_landmark_fields(cases)


def test_invalid_negative_origin_never_identifies_case_duration_or_survival():
    cases = pd.DataFrame({
        "CaseID": [1, 2, 3, 4, 5, 6],
        "ICUOffset": [-3600, 0, 3600, float("inf"), -float("inf"), None],
        "TimeOfStay": [172800] * 6, "OffsetOfDeath": [None] * 6,
        "HospitalDischargeType": [2026] * 6,
    })
    result = sic_case_landmark_fields(cases)
    assert result.loc[[0, 3, 4, 5], "care_unit_to_case_closure_days"].isna().all()
    assert result.loc[[0, 3, 4, 5], "death_during_case"].isna().all()
    assert result.loc[1, "care_unit_to_case_closure_days"] == 2.0
    assert result.loc[2, "care_unit_to_case_closure_days"] == 47 / 24
    assert not result.loc[1, "death_during_case"]
    risk = case_duration_landmark_risk_set(
        duration_hours=result["care_unit_to_case_closure_days"] * 24,
        death_status=result["death_during_case"],
        death_time_hours=result["death_time_during_case_hours"], landmark_hours=24,
    )
    assert risk.eligible.tolist() == [False, True, True, False, False, False]
    assert risk.unidentifiable.tolist() == [True, False, False, True, True, True]


def test_datetime_nat_death_offsets_remain_missing_not_integer_sentinel():
    cases = pd.DataFrame({
        "CaseID": [1, 2], "ICUOffset": [0, 0], "TimeOfStay": [172800, 172800],
        "OffsetOfDeath": pd.to_datetime([None, None]), "HospitalDischargeType": [2026, 2028],
    })
    result = sic_case_landmark_fields(cases)
    assert result["death_during_case"].tolist() == [False, pd.NA]
    assert result["death_time_during_case_hours"].isna().all()


def test_registered_native_callback_uses_only_sic_and_retains_nullable_status():
    from types import SimpleNamespace

    from easyicu.api.extraction import load_dictionary
    from easyicu.concept import ConceptResolver

    dictionary = load_dictionary()
    cases = pd.DataFrame({"CaseID": [1, 2], "ICUOffset": [0, 0], "TimeOfStay": [172800, 172800],
                          "OffsetOfDeath": [None, None], "HospitalDischargeType": [2026, 2028]})
    class DataSource:
        config = SimpleNamespace(name="sic")
        def load_table(self, name, *, columns, filters):
            assert name == "cases"
            assert filters and list(filters[0].value) == [1, 2]
            return SimpleNamespace(data=cases[columns])

    resolver = ConceptResolver(dictionary)
    definition = dictionary["death_during_case"]
    source = definition.sources["sic"][0]
    result = resolver._load_fun_item("death_during_case", definition, source, DataSource(),
                                     aggregator="auto", patient_ids=[1, 2])
    assert result.data["death_during_case"].tolist() == [False, pd.NA]
    mapped = resolver._load_fun_item("death_during_case", definition, source, DataSource(),
                                    aggregator="auto", patient_ids={"CaseID": [1, 2]})
    assert mapped.data["death_during_case"].tolist() == [False, pd.NA]
    with pytest.raises(ValueError, match="one CaseID identifier list"):
        resolver._load_fun_item("death_during_case", definition, source, DataSource(),
                                aggregator="auto", patient_ids={"PatientID": [1, 2]})
    with pytest.raises(ValueError, match="one CaseID identifier list"):
        resolver._load_fun_item("death_during_case", definition, source, DataSource(),
                                aggregator="auto", patient_ids={"CaseID": [1], "caseid": [2]})
    DataSource.config.name = "eicu"
    with pytest.raises(ValueError, match="requires a registered SIC field"):
        resolver._load_fun_item("death_during_case", definition, source, DataSource(),
                                aggregator=False, patient_ids=None)


@pytest.mark.parametrize("status_as_outcome", [False, True])
def test_native_export_seals_nullable_landmark_fields_with_source_axis(tmp_path, status_as_outcome):
    from easyicu.api import extraction as api
    from easyicu.databases.time_origin import SIC_CARE_UNIT_ORIGIN, SourceTimeAxisBinding
    from easyicu.research_agent.cohort.materializer import materialize_cohort
    from easyicu.research_agent.intake.export_package import open_export_package

    concepts = ["care_unit_to_case_closure_days", "death_time_during_case_hours", "death_during_case"]
    assert api.EXTRACT_MODULES["sic_case_landmark"] == concepts
    assert "sic_case_landmark" not in api.EXTRACT_MODULE_ORDER
    pd.DataFrame({"stay_id": [1, 2], concepts[0]: [2.0, 2.0], concepts[1]: [None, None],
                  concepts[2]: pd.Series([False, pd.NA], dtype="boolean")}).to_parquet(
                      tmp_path / "sic_case_landmark.parquet", index=False)
    api._publish_native_export_v2(
        database="sic", data_path="/synthetic-not-read", output_dir=str(tmp_path),
        modules=["sic_case_landmark"], max_patients=None,
        result={"modules": {"sic_case_landmark": {"errors": []}}},
    )
    with open_export_package(tmp_path) as package:
        assert package.declared_time_origin == SIC_CARE_UNIT_ORIGIN
        assert all(concept in package.concept_index for concept in concepts)
        axis = SourceTimeAxisBinding("a" * 64, package.authority_sha256, package.manifest_sha256,
                                     SIC_CARE_UNIT_ORIGIN)
    data = pd.read_parquet(tmp_path / "sic_case_landmark.parquet")
    assert data[concepts[2]].tolist() == [False, pd.NA]
    cohort, receipt = materialize_cohort(
        feature_concepts=(), static_concepts=concepts[:2] if status_as_outcome else concepts,
        outcome_concepts=[concepts[2]] if status_as_outcome else (), database="sic", data_path=tmp_path,
        source_time_axis=axis,
    )
    assert cohort[concepts[2]].iloc[0] == False  # noqa: E712
    assert pd.isna(cohort[concepts[2]].iloc[1])
    assert receipt["source_time_axis"]["manifest_sha256"] == axis.manifest_sha256


def test_landmark_source_snapshot_lists_registered_inputs_only_when_config_verified():
    from easyicu.api.extraction import load_dictionary
    from easyicu.concept.provenance import capture_data_provenance, metadata_digest
    from easyicu.config import load_src_cfg

    dictionary = load_dictionary()
    config = load_src_cfg("sic")
    kwargs = dict(concepts=["care_unit_to_case_closure_days"], source_database="sic")
    recorded = capture_data_provenance(
        dictionary, **kwargs, source_config=config, source_config_sha256=metadata_digest(config.model_dump(mode="json")),
    )
    sources = recorded.concepts[0].sources
    assert {source.value_variable for source in sources} == {
        "CaseID", "ICUOffset", "TimeOfStay", "OffsetOfDeath", "HospitalDischargeType",
    }
    unknown = capture_data_provenance(dictionary, **kwargs)
    assert all(source.value_variable is None for source in unknown.concepts[0].sources)


def test_landmark_module_cannot_hide_unbounded_longitudinal_features():
    from easyicu.research_agent.intake.export_package import ExportPackageError, require_export_time_window_authority

    module = "sic_case_landmark"
    with pytest.raises(ExportPackageError, match="did not limit"):
        require_export_time_window_authority(
            {"database": "sic", "concept_selection": {"modules": {module: ["arbitrary_vitals"]}}}, modules=[module],
        )
    with pytest.raises(ExportPackageError, match="did not limit"):
        require_export_time_window_authority(
            {"database": "eicu", "concept_selection": {"modules": {module: ["arbitrary_vitals"]}}}, modules=[module],
        )


def test_explicit_landmark_module_runs_through_web_export_and_records_source(tmp_path, monkeypatch):
    from contextlib import contextmanager
    from types import SimpleNamespace
    import easyicu.api as api
    from easyicu.config import load_src_cfg
    from easyicu.concept.provenance import record_resolver_inputs
    from easyicu.resources import load_dictionary
    from easyicu.research_agent.intake.export_package import open_export_package
    from easyicu.webserver import dataio
    from easyicu.webserver.time_origin_source_authority import case_duration_source_provenance
    from tests.support.data_provenance import ExportJob

    @contextmanager
    def keep_cache(**_kwargs):
        yield None
    def load(concepts, **kwargs):
        record_resolver_inputs(kwargs, SimpleNamespace(
            concept_resolver=SimpleNamespace(dictionary=load_dictionary()),
            datasource=SimpleNamespace(config=load_src_cfg("sic")),
        ))
        return pd.DataFrame({"stay_id": [1, 2], "care_unit_to_case_closure_days": [2.0, 1.0],
            "death_time_during_case_hours": [None, None], "death_during_case": pd.Series([False, pd.NA], dtype="boolean")})
    monkeypatch.setattr(api, "keep_cache", keep_cache)
    monkeypatch.setattr(api, "load_concepts", load)
    monkeypatch.setattr(api, "get_all_patient_ids", lambda *_, **__: ([1, 2], "stay_id"))
    out = tmp_path / "export"
    result = dataio.make_export_runner(
        data_path=str(tmp_path), database="sic", modules=["sic_case_landmark"],
        export_format="parquet", out_dir=str(out), include_feature_definitions=False,
    )(ExportJob())
    assert result["manifest"]
    with open_export_package(out) as package:
        assert case_duration_source_provenance(package).source_config_sha256
