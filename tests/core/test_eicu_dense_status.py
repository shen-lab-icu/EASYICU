"""Dense source statuses retain unknowns; sparse events keep occurrence semantics."""

from types import SimpleNamespace
import json

import pandas as pd
import pytest

from easyicu.concept import ConceptResolver
from easyicu.concept.schema import ConceptDefinition, ConceptDictionary, ConceptSource
from easyicu.resources import load_data_sources, load_dictionary
from easyicu.table import ICUTable
from easyicu.utils.compat import merge_concepts_r_style


def _source(database):
    frame = pd.DataFrame({
        "patientunitstayid": [1, 1, 2, 2, 3, 3, 4, 4],
        "unitdischargeoffset": [60.0] * 8,
        "hospitaldischargestatus": [None, "Unknown", "Expired", None, "Alive", None, "Alive", "Expired"],
    })
    config = load_data_sources().get(database)

    def load_table(name, **kwargs):
        assert name == "patient"
        return ICUTable(frame.copy(), id_columns=["patientunitstayid"], index_column="unitdischargeoffset")

    return SimpleNamespace(config=config, load_table=load_table, clear_cache=lambda: None)


@pytest.mark.parametrize("database", ["eicu", "eicu_demo"])
@pytest.mark.parametrize("target", ["ts_tbl", "id_tbl"])
@pytest.mark.parametrize("merge", [False, True])
def test_public_status_load_preserves_all_unknown_and_mixed_groups(database, target, merge):
    dictionary = load_dictionary()
    dictionary["death"].target = target
    result = ConceptResolver(dictionary).load_concepts(
        ["death"], _source(database), merge=merge, r_compatible=merge,
        interval=pd.Timedelta(hours=1), align_to_admission=False, verbose=False, concept_workers=1,
    )
    frame = result if merge else result["death"].data
    values = frame.sort_values("patientunitstayid").death.reset_index(drop=True)
    pd.testing.assert_series_equal(values, pd.Series([pd.NA, True, False, True], dtype="boolean", name="death"))


@pytest.mark.parametrize("callback,expected", [
    ("eicu_hospital_mortality_status_v1", [None, True, False]),
    ("identity_callback", [False, True, False]),
])
def test_aggregation_follows_source_binding_even_for_an_alias(callback, expected, monkeypatch):
    # Isolate aggregation from decoding; the public test above exercises both.
    import easyicu.concept as owner

    def decode(frame, *args, **kwargs):
        return frame.assign(endpoint=pd.Series([pd.NA, pd.NA, True, pd.NA, False, pd.NA], dtype="boolean"))

    monkeypatch.setattr(owner, "_apply_callback", decode)
    definition = ConceptDefinition(name="endpoint", target="ts_tbl", sources={"eicu": [ConceptSource(
        table="patient", value_var="endpoint", index_var="unitdischargeoffset", callback=callback,
    )]})
    frame = pd.DataFrame({
        "patientunitstayid": [1, 1, 2, 2, 3, 3], "unitdischargeoffset": [60.0] * 6,
        "endpoint": [0] * 6,
    })
    source = _source("eicu")
    source.load_table = lambda *args, **kwargs: ICUTable(frame.copy(), id_columns=["patientunitstayid"], index_column="unitdischargeoffset")
    result = ConceptResolver(ConceptDictionary({"endpoint": definition})).load_concepts(
        ["endpoint"], source, merge=False, r_compatible=False,
        interval=pd.Timedelta(hours=1), align_to_admission=False, verbose=False, concept_workers=1, _keep_na_rows=True,
    )["endpoint"].data.sort_values("patientunitstayid")
    assert result.endpoint.astype(object).where(result.endpoint.notna(), None).tolist() == expected


@pytest.mark.parametrize("static", [False, True])
def test_dense_merge_retains_missing_values_and_null_time_with_nullable_dtype(static):
    status = pd.DataFrame({
        "stay_id": [1, 1, 2, 2, 3, 3, 4, 5, None],
        "endpoint": pd.Series([pd.NA, pd.NA, True, pd.NA, False, pd.NA, pd.NA, False, True], dtype="boolean"),
        "charttime": [0.0] * 6 + [None, None, 0.0],
    })
    if static:
        status = status.drop(columns="charttime")
    merged = merge_concepts_r_style(
        {"endpoint": status, "measurement": pd.DataFrame({"stay_id": [9], "charttime": [1.0], "measurement": [2.0]})},
        dense_status_columns={"endpoint"},
    )
    assert sorted(merged.stay_id.astype(int)) == [1, 2, 3, 4, 5, 9]
    by_id = merged.set_index("stay_id")
    assert by_id.loc[[1, 4], "endpoint"].isna().all()
    assert by_id.loc[2, "endpoint"] == True  # noqa: E712
    assert by_id.loc[[3, 5], "endpoint"].eq(False).all()
    assert pd.isna(by_id.loc[4, "charttime"])
    assert str(merged.endpoint.dtype) == "boolean"


def test_sparse_boolean_missing_values_still_do_not_create_event_keys():
    merged = merge_concepts_r_style({"event": pd.DataFrame({
        "stay_id": [1, 2, 3], "charttime": [0.0, 0.0, None],
        "event": pd.Series([pd.NA, True, False], dtype="boolean"),
    })})
    assert merged.stay_id.tolist() == [2]


def test_a_dense_status_with_only_unknowns_and_no_clock_is_still_a_record():
    merged = merge_concepts_r_style({"endpoint": pd.DataFrame({
        "stay_id": [1, 1], "charttime": [None, None],
        "endpoint": pd.Series([pd.NA, pd.NA], dtype="boolean"),
    })}, dense_status_columns={"endpoint"})
    assert merged.stay_id.tolist() == [1]
    assert merged.charttime.isna().all() and merged.endpoint.isna().all()
    assert str(merged.endpoint.dtype) == "boolean"


def test_discharge_decoder_recognizes_only_the_two_recorded_statuses():
    from easyicu.hospital_mortality import hospital_mortality_status_from_discharge

    values = pd.Series([" Expired ", "ALIVE", None, "", "Unknown", "Transferred"], index=[8, 7, 6, 5, 4, 3])
    pd.testing.assert_series_equal(
        hospital_mortality_status_from_discharge(values),
        pd.Series([True, False, pd.NA, pd.NA, pd.NA, pd.NA], index=values.index, dtype="boolean"),
    )


def test_binding_overlay_preserves_an_explicit_source_callback():
    definition = ConceptDefinition.from_name_and_payload("death", {"sources": {"eicu": [{
        "table": "patient", "val_var": "hospitaldischargestatus", "callback": "identity_callback",
    }]}})
    assert definition.sources["eicu"][0].callback == "identity_callback"


@pytest.mark.parametrize("dense", [False, True])
def test_recursive_logical_aggregation_obeys_the_source_binding(monkeypatch, dense):
    import easyicu.concept as owner

    definition = ConceptDefinition(
        name="endpoint", callback="synthetic_status", sub_concepts=["raw_status"],
        sources={"eicu": [ConceptSource(callback="eicu_hospital_mortality_status_v1" if dense else "identity_callback")]},
    )
    resolver = ConceptResolver(ConceptDictionary({"endpoint": definition}))
    monkeypatch.setattr(resolver, "load_concepts", lambda *args, **kwargs: {})
    frame = pd.DataFrame({
        "stay_id": [1, 1, 2, 2, 3, 3], "charttime": [0.0] * 6,
        "endpoint": pd.Series([pd.NA, pd.NA, True, pd.NA, False, pd.NA], dtype="boolean"),
    })
    monkeypatch.setattr(owner, "execute_concept_callback", lambda *args: ICUTable(
        frame.copy(), id_columns=["stay_id"], index_column="charttime", value_column="endpoint",
    ))
    result = resolver._load_recursive_concept(
        "endpoint", definition, _source("eicu"), aggregator=None, patient_ids=None,
        verbose=False, interval=pd.Timedelta(hours=1), align_to_admission=False,
    ).data
    expected = [None, True, False] if dense else [False, True, False]
    assert result.endpoint.astype(object).where(result.endpoint.notna(), None).tolist() == expected


@pytest.mark.parametrize("typed", [False, True])
@pytest.mark.parametrize("database", ["eicu", "eicu_demo"])
def test_export_cohort_summary_never_counts_unknown_status_as_survival(tmp_path, typed, database):
    from easyicu.concept.export_metadata import build_export_file_metadata_binding
    from easyicu.hospital_mortality import EICU_HOSPITAL_STATUS_BINDING
    from easyicu.api.extraction import _canonicalise_native_export_frame
    from easyicu.research_agent.cohort.materializer import materialize_cohort
    from easyicu.research_agent.execution.runners.deterministic_descriptive import _group_rows, _reportable_descriptive_results
    from tests.support.native_outcome_export import typed_native_export, untyped_native_export

    extracted = ConceptResolver(load_dictionary()).load_concepts(
        ["death"], _source(database), merge=True, r_compatible=True,
        align_to_admission=False, verbose=False, concept_workers=1,
    ).rename(columns={"patientunitstayid": "stay_id"})
    extracted["los_icu"] = 2.0
    extracted = _canonicalise_native_export_frame(
        extracted, module="outcome", requested_concepts=["death", "los_icu"],
        dictionary=load_dictionary(), database=database,
    )
    binding = build_export_file_metadata_binding(
        relative_path="outcome.parquet", module="outcome", frame=extracted,
        concept_ids=["death", "los_icu"], database=database,
        database_class_prefixes=load_data_sources().get(database).class_prefix,
        dictionary=load_dictionary(),
    )
    lineage, = binding.columns["death"].metadata.source_lineage
    # Semantic parameters are canonical JSON, not unencoded strings. Check
    # the declaration on the same extracted rows whose nulls survive below.
    parameters = {key: json.loads(value) for key, value in lineage.semantic_parameters}
    assert parameters.items() >= {
        "clinical_binding": EICU_HOSPITAL_STATUS_BINDING,
        "event_status": "Expired", "survivor_status": "Alive",
        "unknown_status_policy": "retain_null",
    }.items()
    exporter = typed_native_export if typed else untyped_native_export
    root = exporter(tmp_path / "export", outcome=extracted, outcome_concepts=["death", "los_icu"], database=database)
    assert pd.read_parquet(root / "outcome.parquet").death.isna().sum() == 1
    cohort, _ = materialize_cohort(
        feature_concepts=[], database=database, data_path=str(root),
        cohort_window=(0.0, 24.0), outcome_concepts=["death"], static_concepts=["age"],
    )
    outcomes = cohort.sort_values("stay_id").death.reset_index(drop=True)
    assert outcomes.isna().tolist() == [True, False, False, False]
    assert outcomes.dropna().tolist() == [1, 0, 1]
    table = pd.DataFrame(_group_rows(
        exposure="group", group_type="level", group_value=0, label="All stays",
        mask=pd.Series(True, index=outcomes.index), outcome=outcomes, n_total=4,
        measured_col=None, count_col=None,
    ))
    reporting = _reportable_descriptive_results(table=table, exposures=["group"], context={}, outcome=outcomes, outcome_col="death")
    group = reporting["exposures"][0]["groups"][0]
    assert group["n"] == 4
    assert group["outcome_n"] == 3
    assert group["outcome_missing_n"] == 1
    assert group["outcome_risk_pct"] == pytest.approx(200 / 3)
    assert reporting["overall_outcome"]["missing_n"] == 1


def test_status_decoder_and_export_semantics_share_the_binding(monkeypatch):
    from easyicu import hospital_mortality

    policy = {
        "clinical_binding": hospital_mortality.EICU_HOSPITAL_STATUS_BINDING,
        "event_status": "Synthetic event", "survivor_status": "Synthetic survivor",
        "unknown_status_policy": "retain_null",
    }
    monkeypatch.setattr(hospital_mortality, "EICU_HOSPITAL_STATUS_PARAMETERS", policy, raising=False)
    definition = ConceptDefinition.from_name_and_payload("death", {"sources": {"eicu": [{
        "table": "patient", "val_var": "hospitaldischargestatus",
        "callback": "transform_fun(comp_na(`==`, 'Expired'))",
    }]}})
    assert definition.sources["eicu"][0].params.items() >= policy.items()
    actual = hospital_mortality.hospital_mortality_status_from_discharge(
        pd.Series(["Synthetic event", "Synthetic survivor", "Unknown"])
    )
    pd.testing.assert_series_equal(actual, pd.Series([True, False, pd.NA], dtype="boolean"))


@pytest.mark.parametrize("database", ["eicu", "eicu_demo"])
def test_status_overlay_carries_the_declared_unknown_policy_into_export_lineage(database):
    from easyicu.concept.export_metadata import build_export_file_metadata_binding
    from easyicu.hospital_mortality import EICU_HOSPITAL_STATUS_BINDING

    dictionary = load_dictionary()
    source, = dictionary["death"].for_data_source(load_data_sources().get(database))
    expected = {
        "clinical_binding": EICU_HOSPITAL_STATUS_BINDING,
        "event_status": "Expired", "survivor_status": "Alive",
        "unknown_status_policy": "retain_null",
    }
    assert all(source.params.get(key) == value for key, value in expected.items())
    frame = pd.DataFrame({
        "patientunitstayid": [1, 2, 3],
        "death": pd.Series([True, False, pd.NA], dtype="boolean"),
    })
    binding = build_export_file_metadata_binding(
        relative_path="outcome.parquet", module="outcome", frame=frame,
        concept_ids=["death"], database=database,
        database_class_prefixes=load_data_sources().get(database).class_prefix,
        dictionary=dictionary,
    )
    lineage, = binding.columns["death"].metadata.source_lineage
    assert dict(lineage.semantic_parameters)["unknown_status_policy"] == '"retain_null"'
    assert lineage.callback == EICU_HOSPITAL_STATUS_BINDING
