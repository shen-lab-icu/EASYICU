"""Landed R6 review: synthetic admissibility and real-column matching edges."""

from __future__ import annotations

import pandas as pd
import pytest

from easyicu.concept import ConceptResolver
from easyicu.concept.emar_diuretics import drop_inputevent_duplicates
from easyicu.concept.loader import load_dictionary
from tests.support.eicu_source import _eicu_source


@pytest.mark.parametrize("database", ["eicu", "eicu_demo"])
@pytest.mark.parametrize("layout", ["file", "bucket", "table"])
@pytest.mark.parametrize("name,label,value", [
    ("ph", "pH", 7.4), ("sao2", "O2 Sat (%)", 98.),
    ("o2sat", "O2 Sat (%)", 98.), ("be", "Base Deficit", 2.),
])
def test_rr_01_only_admitted_pressure_results_anchor_other_gases(
    tmp_path, database, layout, name, label, value,
):
    # An out-of-bounds, infinite or missing pressure is not an admitted paO2/
    # paCO2. A valid result of either analyte is sufficient, on the exact clock.
    pressures = [("paO2", v) for v in (0., -1., float("inf"), 601., 19., None, 80.)]
    pressures += [("paCO2", v) for v in (0., 151., 9., 40.)]
    lab = pd.DataFrame([
        (analyte, result, i * 60)
        for i, (gas, result) in enumerate(pressures)
        for analyte, result in ((gas, result), (label, value), ("lactate", 1.), ("sodium", 140.))
    ], columns=["labname", "labresult", "labresultoffset"]).assign(labmeasurenameinterface="")
    # Explicit specimens still decide first, even without a pressure anchor.
    samples = pd.DataFrame({"labotheroffset": [0, 600], "labothername": "Sample Type",
                            "labothervaluetext": ["Arterial", "Venous"], "labotherresult": ""})
    source = _eicu_source(tmp_path, database, lab=lab, customlab=samples)
    if layout == "bucket":
        directory = tmp_path / "lab_bucket" / "bucket_id=0"
        directory.mkdir(parents=True)
        (directory.parent / "_COMPLETE").write_text("ready")
        (tmp_path / "lab.parquet").rename(directory / "part.parquet")
    if layout == "table":
        source.resolve_bucket_directory = lambda _: None
        source.resolve_flat_parquet_directory = lambda _: None
    names = ["po2", "pco2", name, "lact", "na"]
    resolver = ConceptResolver(load_dictionary())
    kwargs = dict(interval=pd.Timedelta(hours=1), concept_workers=1, verbose=False)
    single = resolver.load_concepts([name], source, merge=False, r_compatible=False, **kwargs)[name]
    expected = {0.: -value if name == "be" else value, 6.: -value if name == "be" else value}
    actual = dict(zip(single.data[single.index_column], single.data[name]))
    assert actual == expected
    if name != "o2sat":
        assert single.data[f"{name}_specimen_status"].tolist() == ["explicit_arterial", "co_measured_arterial"]
    # Four or more concepts on a shared long table exercise the partial-wide
    # merge, including two ordinary lab concepts that can use the SQL batch.
    for compatible in (False, True):
        merged = ConceptResolver(load_dictionary()).load_concepts(
            names, source, merge=True, r_compatible=compatible, **kwargs,
        )
        got = merged.loc[merged[name].notna()].set_index("charttime").sort_index()
        assert got[name].to_dict() == pytest.approx(expected)
        if name != "o2sat":
            assert got[f"{name}_specimen_status"].tolist() == ["explicit_arterial", "co_measured_arterial"]


def _partial_panel(tmp_path, database, layout):
    from tests.support.blood_gas_sources import _source, _write, _mimic_family

    names = ["po2", "ph", "lact", "na"]
    values = [80., 7.4, 1., 140.]
    dictionary = load_dictionary()
    ids = {name: dictionary[name].sources[database][0].ids[0] for name in names}
    if database.startswith("eicu"):
        source = _eicu_source(tmp_path, database, lab=pd.DataFrame([
            (ids[name], value, offset) for offset in (0, 60)
            for name, value in zip(names, values)
        ], columns=["labname", "labresult", "labresultoffset"]).assign(labmeasurenameinterface=""))
        table = "lab"
    elif database.startswith(("miiv", "mimic")):
        miiv = database == "miiv"
        rows = [(minute, ids[name], str(value), panel) for panel, minute in ((1, 0), (2, 60))
                for name, value in zip(names, values)]
        rows += [(minute, 52033 if miiv else 50800, "ART." if miiv else "ART", panel)
                 for panel, minute in ((1, 0), (2, 60))]
        source, _ = _mimic_family(tmp_path, database, rows, [], "specimen_id" if miiv else "unused")
        table = "labevents"
    elif database == "aumc":
        from tests.support.aumc_source import aumc_db

        source = aumc_db(tmp_path, numeric=[
            (minute, ids[name], value, "mmHg" if name == "po2" else "")
            for minute in (5, 65) for name, value in zip(names, values)
        ], texts=[(5, 11646, "ART."), (65, 11646, "ART.")])
        table = "numericitems"
    elif database == "hirid":
        from tests.support.hirid_source import hirid_source_root

        root = hirid_source_root(tmp_path, "observations", [
            {"patientid": 1, "datetime": hour, "variableid": ids[name], "value": value}
            for hour in (.1, 1.1) for name, value in zip(names, values)
        ])
        source = _source(root, database)
        table = "observations"
    else:
        _write(pd.DataFrame({"CaseID": [1], "PatientID": [11], "ICUOffset": [0], "TimeOfStay": [172800]}),
               tmp_path / "cases.parquet")
        rows = [(offset, ids[name], value) for offset in (7500, 11100) for name, value in zip(names, values)]
        frame = pd.DataFrame(rows, columns=["Offset", "LaboratoryID", "LaboratoryValue"]).assign(CaseID=1, LaboratoryType=2296)
        frame["id"] = range(len(frame))
        _write(frame, tmp_path / "laboratory.parquet")
        source = _source(tmp_path, database)
        table = "laboratory"
    root = source.base_path
    flat = root / table
    if not flat.is_dir():
        flat.mkdir()
        (root / f"{table}.parquet").rename(flat / "part.parquet")
    if layout == "bucket":
        directory = root / f"{table}_bucket" / "bucket_id=0"
        directory.mkdir(parents=True)
        for file in flat.glob("*.parquet"):
            file.rename(directory / file.name)
        (directory.parent / "_COMPLETE").write_text("ready")
    return source


@pytest.mark.parametrize("database", ["eicu", "eicu_demo", "aumc", "hirid", "miiv", "mimic", "mimic_demo", "sic"])
@pytest.mark.parametrize("layout", ["flat", "bucket"])
def test_rr_01_partial_batch_keeps_gases_companions_and_ordinary_labs(tmp_path, monkeypatch, database, layout):
    from easyicu import datasource

    source = _partial_panel(tmp_path, database, layout)
    batches = []
    original = datasource.load_bucketed_table_multi_aggregated

    def record_batch(*args, **kwargs):
        result = original(*args, **kwargs)
        batches.append(set(args[2]))
        return result

    monkeypatch.setattr(datasource, "load_bucketed_table_multi_aggregated", record_batch)
    names = ["po2", "ph", "lact", "na"]
    for compatible in (False, True):
        result = ConceptResolver(load_dictionary()).load_concepts(
            names, source, merge=True, r_compatible=compatible, interval=pd.Timedelta(hours=1),
            concept_workers=1, verbose=False,
        )
        clock = next(column for column in ("charttime", "measuredat", "datetime", "Offset") if column in result)
        result = result.sort_values(clock)
        assert result[clock].tolist() == ([2., 3.] if database == "sic" else [0., 1.])
        for name, expected in zip(names, (80., 7.4, 1., 140.)):
            assert result[name].tolist() == pytest.approx([expected, expected])
        po2_status = "label_defined_arterial" if database.startswith("eicu") or database == "hirid" else "explicit_arterial"
        ph_status = "co_measured_arterial" if database.startswith("eicu") else po2_status
        assert result.po2_specimen_status.tolist() == [po2_status] * 2
        assert result.ph_specimen_status.tolist() == [ph_status] * 2
    assert batches == [{"lact", "na"}] * 2


@pytest.mark.parametrize("drug,item", [("furosemide", 221794), ("bumetanide", 229639)])
@pytest.mark.parametrize("input_minutes", [[9, 20], [40, 100]])
def test_rr_07_repeated_doses_use_each_matching_input_once(drug, item, input_minutes):
    t = pd.Timestamp("2150-01-01")
    emar_minutes = [0, 10] if input_minutes == [9, 20] else [0, 50]
    doses = pd.DataFrame({"hadm_id": 1, "charttime": t + pd.to_timedelta(emar_minutes + [0, 0, 180], unit="m"),
                          "_route": ["iv", "iv", "oral", None, "iv"], "_dose": 40., "_unit": "mg"})
    inputs = pd.DataFrame({"hadm_id": 1, "itemid": item, "amount": 40., "amountuom": "mg",
                           "starttime": t + pd.to_timedelta(input_minutes, unit="m")})
    # Both IV administrations have distinct input copies. A nearest-only join
    # points both at the first input and leaves one duplicate in the export.
    for reverse in (False, True):
        left = doses.iloc[::-1] if reverse else doses
        right = inputs.iloc[::-1] if reverse else inputs
        kept = drop_inputevent_duplicates(left, right, key="hadm_id", time_col="charttime", drug=drug)
        assert set(kept.index) == {2, 3, 4}
    # One input still removes at most one IV dose, and never an oral dose.
    kept = drop_inputevent_duplicates(doses, inputs.iloc[:1], key="hadm_id", time_col="charttime", drug=drug)
    assert len(kept) == 4
    assert {2, 3, 4} <= set(kept.index)


@pytest.mark.parametrize("bucketed", [False, True])
def test_rr_07_repeated_real_emar_rows_are_counted_once_in_complete_concept(tmp_path, monkeypatch, bucketed):
    import os
    import tempfile

    from easyicu.api import extraction
    from easyicu.api.concepts import clear_global_loader
    from tests.support.mimic_synthetic_source import mimic_synthetic_source

    monkeypatch.setattr(tempfile, "tempdir", tempfile.tempdir)
    for variable in ("TMPDIR", "EASYICU_DUCKDB_TEMP_DIR"):
        if variable in os.environ:
            monkeypatch.setenv(variable, os.environ[variable])
        else:
            monkeypatch.delenv(variable, raising=False)
    t = pd.Timestamp("2150-01-01")
    source = mimic_synthetic_source(
        tmp_path, "miiv", bucketed=bucketed,
        emar=pd.DataFrame({"subject_id": 11, "hadm_id": 21, "emar_id": ["a", "b", "oral"],
                           "pharmacy_id": [1, 2, 3], "medication": "Furosemide",
                           "charttime": t + pd.to_timedelta([0, 10, 180], unit="m"), "event_txt": "Administered"}),
        emar_detail=pd.DataFrame({"emar_id": ["a", "b", "oral"], "parent_field_ordinal": 1.,
                                  "dose_given": "40", "dose_given_unit": "mg", "route": ["IV", "IV", "PO"]}),
        prescriptions=pd.DataFrame({"pharmacy_id": [1, 2, 3], "route": ["IV", "IV", "PO"]}),
        inputevents=pd.DataFrame({"subject_id": 11, "hadm_id": 21, "stay_id": 1, "orderid": [1, 2],
                                  "linkorderid": [1, 2], "itemid": 221794, "amount": 40., "amountuom": "mg",
                                  "starttime": t + pd.to_timedelta([9, 20], unit="m"),
                                  "endtime": t + pd.to_timedelta([10, 21], unit="m"),
                                  "statusdescription": "FinishedRunning", "rate": 0., "rateuom": "mg/hour"}),
    )
    # No fabricated administration ID is present in inputevents. Read all
    # sources together, including the surviving input and the separate PO dose.
    name = "loop_diuretic_furosemide_eq"
    for merge, compatible in ((False, False), (True, False), (True, True)):
        loaded = ConceptResolver(load_dictionary()).load_concepts(
            [name], source, merge=merge, r_compatible=compatible, interval=pd.Timedelta(hours=1),
            patient_ids={"stay_id": [1]}, concept_workers=1, verbose=False,
        )
        frame = loaded if merge else loaded[name].data
        clock = "charttime" if merge else loaded[name].index_column
        assert frame.loc[frame[name].notna()].set_index(clock)[name].to_dict() == {0.: 80., 3.: 20.}
    monkeypatch.setitem(extraction.EXTRACT_MODULES, "medications", [name])
    monkeypatch.setitem(extraction.EXTRACT_MODULES, "outcome", ["los_icu"])
    try:
        result = extraction.extract_database(
            "miiv", data_path=str(tmp_path), output_dir=str(tmp_path / "native"),
            modules=["medications", "outcome"], native_export_v2=True,
            verbose=False, resource_budget_mb=1024,
        )
        assert not {module: state["errors"] for module, state in result["modules"].items() if state.get("errors")}
        published = pd.read_parquet(tmp_path / "native/medications.parquet")
        kept = published.loc[published[name].notna()]
        assert kept.set_index("charttime")[name].to_dict() == {0.: 80., 3.: 20.}
    finally:
        clear_global_loader()


def test_rr_07_matching_skips_exhausted_windows_without_stealing_a_later_copy():
    t = pd.Timestamp("2150-01-01")
    doses = pd.DataFrame({"hadm_id": [1] * 4 + [2], "_route": "iv", "_dose": 40., "_unit": "mg",
                          "charttime": t + pd.to_timedelta([0, 0, 60, 120, 0], unit="m")})
    inputs = pd.DataFrame({"hadm_id": 1, "itemid": 221794, "amount": 40., "amountuom": "mg",
                           "starttime": t + pd.to_timedelta([0, 70, 180], unit="m")})
    kept = drop_inputevent_duplicates(doses, inputs, key="hadm_id", time_col="charttime", drug="furosemide")
    assert kept.index.tolist() == [1, 4]


@pytest.mark.parametrize("minutes,input_minute,kept", [([30, 80], 75, [0]), ([0, 45, 80], 50, [0, 2])])
def test_rr_07_single_input_preserves_nearest_copy_and_surviving_hour(minutes, input_minute, kept):
    t = pd.Timestamp("2150-01-01")
    doses = pd.DataFrame({"hadm_id": 1, "_route": "iv", "_dose": 40., "_unit": "mg",
                          "charttime": t + pd.to_timedelta(minutes, unit="m")})
    inputs = pd.DataFrame({"hadm_id": [1], "itemid": 221794, "amount": 40., "amountuom": "mg",
                           "starttime": t + pd.Timedelta(minutes=input_minute)})
    actual = drop_inputevent_duplicates(doses, inputs, key="hadm_id", time_col="charttime", drug="furosemide")
    assert actual.index.tolist() == kept


@pytest.mark.parametrize("database", ["eicu", "eicu_demo"])
@pytest.mark.parametrize("bucketed", [False, True])
def test_rr_01_rr_04_native_modules_keep_specimen_and_medication_evidence(tmp_path, monkeypatch, database, bucketed):
    import os
    import tempfile

    from easyicu.api import extraction
    from easyicu.api.concepts import clear_global_loader

    monkeypatch.setattr(tempfile, "tempdir", tempfile.tempdir)
    for variable in ("TMPDIR", "EASYICU_DUCKDB_TEMP_DIR"):
        if variable in os.environ:
            monkeypatch.setenv(variable, os.environ[variable])
        else:
            monkeypatch.delenv(variable, raising=False)
    gas_names = ["po2", "ph", "be", "sao2", "lact", "na"]
    drugs = ["heparin", "midazolam", "nicardipine", "aspirin"]
    lab = pd.DataFrame([
        (label, value, offset)
        for offset, pressure in ((0, 0.), (60, 80.), (120, 90.), (180, 50.))
        for label, value in (("paO2", pressure), ("pH", 7.4), ("Base Excess", 2.),
                             ("O2 Sat (%)", 98.), ("lactate", 1.), ("sodium", 140.))
    ], columns=["labname", "labresult", "labresultoffset"]).assign(labmeasurenameinterface="")
    medication = pd.DataFrame([
        (drug, i * 60, dose, route)
        for drug in drugs
        for i, (dose, route) in enumerate((("2 mg", "IV"), ("0 mg", "IV"), (None, "IV"),
                                           ("2 mg", "PO"), ("2 mg", None), ("2 mg", "IV")))
    ], columns=["drugname", "drugstartoffset", "dosage", "routeadmin"]).assign(
        drughiclseqno=-1, drugordercancelled="No",
    )
    samples = pd.DataFrame({"labotheroffset": [120, 180], "labothername": "Sample Type",
                            "labothervaluetext": ["Arterial", "Venous"], "labotherresult": ""})
    source = _eicu_source(tmp_path / "source", database, lab=lab, customlab=samples, medication=medication)
    if bucketed:
        for table in ("lab", "medication", "customlab"):
            directory = source.base_path / f"{table}_bucket" / "bucket_id=0"
            directory.mkdir(parents=True)
            (directory.parent / "_COMPLETE").write_text("ready")
            (source.base_path / f"{table}.parquet").rename(directory / "part.parquet")
    for compatible in (False, True):
        merged = ConceptResolver(load_dictionary()).load_concepts(
            gas_names + drugs, source, merge=True, r_compatible=compatible,
            interval=pd.Timedelta(hours=1), concept_workers=1, verbose=False,
        )
        for name in drugs:
            positive = merged.loc[merged[name].eq(True), "charttime"]  # noqa: E712
            assert set(positive) == ({0., 3., 4., 5.} if name == "aspirin" else {0., 5.})
            assert not bool(merged.loc[merged.charttime.eq(1), name].iloc[0])
        for name in drugs[:-1]:
            assert merged.loc[merged.charttime.isin([2, 4]), name].isna().all()
    for module, names in (("blood_gas", gas_names), ("medications", drugs), ("outcome", ["los_icu"])):
        monkeypatch.setitem(extraction.EXTRACT_MODULES, module, names)
    try:
        result = extraction.extract_database(
            database, data_path=str(source.base_path), output_dir=str(tmp_path / "native"),
            modules=["blood_gas", "medications", "outcome"], native_export_v2=True,
            verbose=False, resource_budget_mb=1024,
        )
        assert not {module: state["errors"] for module, state in result["modules"].items() if state.get("errors")}
        gases = pd.read_parquet(tmp_path / "native/blood_gas.parquet").sort_values("charttime")
        for name in ("ph", "be", "sao2"):
            kept = gases.loc[gases[name].notna()]
            assert kept.charttime.tolist() == [1., 2.]
            assert kept[f"{name}_specimen_status"].tolist() == ["co_measured_arterial", "explicit_arterial"]
        meds = pd.read_parquet(tmp_path / "native/medications.parquet")
        for name in drugs:
            positive = meds.loc[meds[name].eq(True), "charttime"]  # noqa: E712
            assert set(positive) == ({0., 3., 4., 5.} if name == "aspirin" else {0., 5.})
    finally:
        clear_global_loader()
