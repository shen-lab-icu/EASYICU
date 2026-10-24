"""Sparse invented sources publish honest availability through the web runner."""

from __future__ import annotations

import csv
import json
from types import SimpleNamespace

import pandas as pd
import pytest

from easyicu.concept.callbacks import (
    ConceptCallbackContext,
    _callback_sep3,
    _callback_sep3_sofa2,
    _callback_supp_o2,
    _callback_susp_inf,
)
from easyicu.table import ICUTable
from easyicu.webserver import dataio
from easyicu.webserver.demo_source_contracts import SOURCES
from tests.support.windows_laptop_demo import isolated_demo_paths, synthetic_demo_download


class Job:
    def emit(self, _payload):
        pass


def _converted_source(tmp_path, database, *, populated=False, medication="vancomycin"):
    source = next(source for source in SOURCES if source.database.removesuffix("_demo") == database)
    paths = isolated_demo_paths(tmp_path, source)
    synthetic_demo_download(source, paths, Job())
    if populated:
        for table, rows in {
            "medication": [
                {"medicationid": index, "patientunitstayid": 101, "drugname": medication,
                 "drugstartoffset": offset, "drugstopoffset": offset + 60, "drugorderoffset": offset}
                for index, offset in enumerate((60, 180), 1)
            ],
            "diagnosis": [
                {"diagnosisid": 1, "patientunitstayid": 101, "diagnosisoffset": 30,
                 "diagnosisstring": "infection", "icd9code": "038.9"},
            ],
        }.items():
            path = paths.raw / f"{table}.csv"
            with path.open(newline="") as handle:
                columns = next(csv.reader(handle))
            with path.open("a", newline="") as handle:
                csv.DictWriter(handle, fieldnames=columns).writerows(rows)
    result = dataio.make_convert_runner(str(paths.raw), database)(Job())
    assert result["failed"] == 0
    return paths


def _export(paths, database, modules):
    result = dataio.make_export_runner(
        str(paths.raw), database, modules=modules, out_dir=str(paths.export),
        export_format="parquet", include_feature_definitions=False,
    )(Job())
    assert result["manifest"] == "_manifest.json"
    return json.loads((paths.export / result["manifest"]).read_text())


@pytest.mark.parametrize("missing_tables", [(), ("medication", "infusiondrug"), ("medication",)])
def test_empty_antibiotics_are_evaluated_but_absent_inputs_are_unavailable(tmp_path, missing_tables):
    paths = _converted_source(tmp_path, "eicu")
    for table in missing_tables:
        # Only this test's invented raw/converted table files are removed.
        for path in paths.raw.glob(f"{table}.*"):
            path.unlink()
    manifest = _export(paths, "eicu", ["sepsis_shared", "sepsis3_sofa1", "sepsis3_sofa2"])
    entries = {entry["module"]: entry for entry in manifest["files"]}
    for module, concept in (
        ("sepsis_shared", "susp_inf"),
        ("sepsis3_sofa1", "sep3_sofa1"),
        ("sepsis3_sofa2", "sep3_sofa2"),
    ):
        status = entries[module]["concept_status"][concept]
        assert status["non_null"] == 0
        if missing_tables:
            assert status["availability"] == "structurally_unavailable_placeholder"
            assert status["evaluation_status"] == "not_evaluable"
            assert status["reason_code"] == "required_source_table_missing"
            assert set(status["missing_tables"]) == set(missing_tables)
        else:
            assert status["availability"] == "produced_all_null"
            assert status["evaluation_status"] == "evaluated"
            assert status["reason_code"] == "no_antibiotics_observed"
        frame = pd.read_parquet(paths.export / entries[module]["file"])
        assert "patientunitstayid" in frame
        assert "stay_id" not in frame
        assert not frame[concept].fillna(False).any()


def test_header_only_labs_publish_an_empty_module_with_primary_bindings(tmp_path):
    from easyicu.research_agent.intake.export_package import concepts_without_values, open_export_package

    paths = _converted_source(tmp_path, "miiv")
    assert pd.read_csv(paths.raw / "labevents.csv").empty
    manifest = _export(paths, "miiv", ["chemistry"])
    entry = manifest["files"][0]
    assert entry["rows"] == 0
    assert entry["concept_status"]["crea"] == {
        "availability": "produced_all_null", "non_null": 0, "reason_code": "data_missing",
    }
    assert "crea" in concepts_without_values(paths.export)
    with open_export_package(paths.export) as package:
        assert "crea" in package.concept_index


def test_sparse_special_scores_are_placeholders_instead_of_negative_scores(tmp_path):
    paths = _converted_source(tmp_path, "miiv")
    manifest = _export(paths, "miiv", ["circulatory"])
    entry = manifest["files"][0]
    for concept in ("circ_event", "circ_failure"):
        assert entry["concept_status"][concept] == {
            "availability": "structurally_unavailable_placeholder", "non_null": 0,
            "reason_code": "producer_returned_no_physical_output", "evaluation_status": "not_evaluable",
        }
        assert pd.read_parquet(paths.export / entry["file"])[concept].isna().all()


@pytest.mark.parametrize("component", ["missing", "empty_without_time", "populated_without_time"])
def test_suspicion_component_gaps_return_typed_receipts(component):
    tables = {
        "infection_icd": ICUTable(
            pd.DataFrame({"stay_id": [7], "infection_icd": [True]}),
            id_columns=["stay_id"], value_column="infection_icd",
        ),
    }
    if component != "missing":
        tables["abx"] = ICUTable(
            pd.DataFrame({"stay_id": [7], "abx": [True]}).iloc[
                :0 if component == "empty_without_time" else 1
            ], id_columns=["stay_id"], value_column="abx",
        )
    context = ConceptCallbackContext(
        "susp_inf", None, pd.Timedelta(hours=1), None,
        SimpleNamespace(config=SimpleNamespace(name="eicu")), None,
    )
    result = _callback_susp_inf(tables, context)
    assert result.data.empty
    assert result.index_column == "charttime"
    receipt = result.data.attrs["easyicu_availability"]
    assert receipt["evaluation_status"] == (
        "evaluated" if component == "empty_without_time" else "not_evaluable"
    )
    assert receipt["reason_code"] == {
        "missing": "required_component_missing",
        "empty_without_time": "no_antibiotics_observed",
        "populated_without_time": "antibiotic_time_index_missing",
    }[component]


@pytest.mark.parametrize("mode", ["and", "or", "abx", "samp"])
def test_empty_antibiotics_preserve_sampling_only_and_or_modes(mode):
    tables = {
        "abx": ICUTable(pd.DataFrame(columns=["stay_id", "abx"]), ["stay_id"], value_column="abx"),
        "samp": ICUTable(
            pd.DataFrame({"stay_id": [7], "charttime": [2.0], "samp": [True]}),
            ["stay_id"], "charttime", "samp",
        ),
    }
    context = ConceptCallbackContext(
        "susp_inf", None, pd.Timedelta(hours=1), None,
        SimpleNamespace(config=SimpleNamespace(name="miiv")), None, {"si_mode": mode},
    )
    result = _callback_susp_inf(tables, context)
    assert result.data["susp_inf"].tolist() == ([True] if mode in {"or", "samp"} else [])


def test_nonempty_components_without_time_are_typed_unavailable():
    tables = {
        name: ICUTable(pd.DataFrame({"stay_id": [7], name: [True]}), ["stay_id"], value_column=name)
        for name in ("abx", "samp")
    }
    context = ConceptCallbackContext(
        "susp_inf", None, pd.Timedelta(hours=1), None,
        SimpleNamespace(config=SimpleNamespace(name="miiv")), None,
    )
    result = _callback_susp_inf(tables, context)
    receipt = result.data.attrs["easyicu_availability"]
    assert receipt["evaluation_status"] == "not_evaluable"
    assert receipt["reason_code"] == "component_time_index_missing"


@pytest.mark.parametrize("concept,callback", [("sep3", _callback_sep3), ("sep3_sofa2", _callback_sep3_sofa2)])
def test_sepsis_without_required_components_is_not_evaluable(concept, callback):
    context = ConceptCallbackContext(
        concept, None, pd.Timedelta(hours=1), None,
        SimpleNamespace(config=SimpleNamespace(name="miiv")), None,
    )
    result = callback({}, context)
    assert result.data.empty
    receipt = result.data.attrs["easyicu_availability"]
    assert receipt["evaluation_status"] == "not_evaluable"
    assert receipt["reason_code"] == "required_component_missing"


@pytest.mark.parametrize("wide_column_batch_size", [2, 8])
def test_populated_source_keeps_antibiotic_owned_infection_times(tmp_path, monkeypatch, wide_column_batch_size):
    import easyicu.api as api
    from easyicu.concept.catalog import CONCEPT_GROUPS_INTERNAL

    # Exercise both constrained hosts' column chunks and the unsplit query
    # used on larger hosts; both must publish the same physical key order.
    monkeypatch.setenv("EASYICU_WIDE_COLUMN_BATCH_SIZE", str(wide_column_batch_size))
    paths = _converted_source(tmp_path, "eicu", populated=True)
    manifest = _export(paths, "eicu", ["vitals", "sofa1_score", "sofa2_score", "sepsis_shared", "sepsis3_sofa1", "sepsis3_sofa2"])
    entries = {entry["module"]: entry for entry in manifest["files"]}
    suspicion = pd.read_parquet(paths.export / entries["sepsis_shared"]["file"])
    observed = suspicion.loc[suspicion["susp_inf"].eq(True), ["patientunitstayid", "charttime"]]
    assert observed.to_dict("list") == {"patientunitstayid": [101, 101], "charttime": [1.0, 3.0]}
    assert entries["sepsis_shared"]["concept_status"]["susp_inf"]["availability"] == "available"
    for module, concept in (("vitals", "hr"), ("sofa1_score", "sofa"), ("sofa2_score", "sofa2")):
        assert entries[module]["concept_status"][concept]["availability"] == "available"
        assert entries[module]["concept_status"][concept]["non_null"] > 0
        frame = pd.read_parquet(paths.export / entries[module]["file"])
        assert frame[["patientunitstayid", "charttime"]].to_dict("list") == {
            "patientunitstayid": [101, 102], "charttime": [1.0, 1.0],
        }, module
    # Match the populated loader output without availability collection. Compare
    # physical order as well as values/dtypes; sorting columns hides regressions.
    with api.keep_cache(database="eicu", data_path=str(paths.raw)):
        for module, entry in entries.items():
            plan = dataio.compile_concept_load_plan(CONCEPT_GROUPS_INTERNAL[module])
            expected = api.load_concepts(
                list(plan.source_concepts), database="eicu", data_path=str(paths.raw),
                patient_ids={"patientunitstayid": [101, 102]}, batch_size=2,
                use_sofa2=module in {"sofa2_score", "sepsis3_sofa2"},
                merge=True, verbose=False,
            )
            expected = dataio._materialize_concept_load_plan(expected, plan)
            actual = pd.read_parquet(paths.export / entry["file"])
            pd.testing.assert_frame_equal(actual, expected, obj=module)


@pytest.mark.parametrize("module", [
    "circulatory", "other_scores", "outcome", "renal", "sepsis_shared", "chemistry",
])
def test_availability_preserves_full_module_column_order(tmp_path, module):
    from easyicu.concept.catalog import CONCEPT_GROUPS_INTERNAL
    from easyicu.resources import load_dictionary

    plan = dataio.compile_concept_load_plan(CONCEPT_GROUPS_INTERNAL[module])
    # A full producer result need not follow the catalog/template order.
    frame = pd.DataFrame({
        "patientunitstayid": [101], "charttime": [1.0],
        **{concept: [1.0] for concept in reversed(plan.output_concepts)},
    })
    expected = frame.copy()
    prepared, statuses = dataio._prepare_export_availability(
        frame, module=module, plan=plan, records={},
        dictionary=load_dictionary(include_sofa2=True),
    )
    output = tmp_path / f"{module}.parquet"
    prepared.to_parquet(output, index=False)
    pd.testing.assert_frame_equal(pd.read_parquet(output), expected)
    assert all(status["availability"] == "available" for status in statuses.values())


def test_non_antibiotic_medications_do_not_make_infection_positive(tmp_path):
    paths = _converted_source(tmp_path, "eicu", populated=True, medication="acetaminophen")
    manifest = _export(paths, "eicu", ["sepsis_shared"])
    status = manifest["files"][0]["concept_status"]["susp_inf"]
    assert status["availability"] == "produced_all_null"
    assert status["reason_code"] == "no_antibiotics_observed"
    assert status["evaluation_status"] == "evaluated"


@pytest.mark.parametrize("populated", [None, "fio2", "vent_ind"])
def test_supplemental_oxygen_handles_empty_time_axes(populated):
    tables = {}
    for name in ("fio2", "vent_ind"):
        frame = pd.DataFrame({
            "stay_id": [7], "charttime": [pd.Timestamp("2100-01-01")],
            name: [40.0 if name == "fio2" else True],
        })
        if populated != name:
            frame = frame.iloc[:0].copy()
            if name == "vent_ind":
                frame["charttime"] = pd.Series(dtype="object")
        tables[name] = ICUTable(frame, ["stay_id"], "charttime", name)
    result = _callback_supp_o2(tables, None)
    assert result.data["supp_o2"].tolist() == ([] if populated is None else [True])
