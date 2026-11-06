"""Round-5b R6 regressions; every clinical row is a synthetic literal."""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import easyicu
from easyicu.concept import ConceptDictionary, ConceptResolver
from easyicu.concept.callback_apply import _apply_callback
from easyicu.concept.schema import ConceptSource
from easyicu.datasource import MissingTableSourceError
from tests.support.eicu_source import _eicu_source


DATA = Path(easyicu.__file__).parent / "data"
IV_DRUGS = {
    "albumin_iv", "dexmedetomidine", "dextrose50", "heparin", "mannitol",
    "midazolam", "milrinone", "nicardipine", "nitroglycerin", "propofol",
    "rocuronium",
}
DRUG_NAMES = {
    "abx": "Meropenem", "albumin_iv": "Albumin human 25%",
    "aspirin": "Aspirin", "cisatracurium": "Cisatracurium",
    "cort": "Hydrocortisone", "dexamethasone": "Dexamethasone",
    "dexmedetomidine": "Dexmedetomidine", "dextrose50": "Dextrose 50%",
    "enoxaparin": "Enoxaparin", "furosemide": "Furosemide",
    "heparin": "Heparin", "lorazepam": "Lorazepam", "mannitol": "Mannitol",
    "meropenem": "Meropenem", "midazolam": "Midazolam", "milrinone": "Milrinone",
    "neostigmine": "Neostigmine", "nicardipine": "Nicardipine",
    "nitroglycerin": "Nitroglycerin IV infusion", "phenytoin": "Phenytoin",
    "propofol": "Propofol", "rocuronium": "Rocuronium", "vancomycin": "Vancomycin",
    "warfarin": "Warfarin", "loop_diuretic": "Furosemide",
    "thiazide_diuretic": "Metolazone", "potassium_sparing_diuretic": "Spironolactone",
    "acetazolamide": "Acetazolamide", "delirium_tx_proxy": "Haloperidol",
}
# Each order has its own hour; a name containing a strength cannot replace dosage.
DOSES = ["2 mg", "0 mg", None, "", "-1 mg", "held", ".5 mg", "0.5 units",
         "2 mg", "2 mg", "2 mg", "2 mg", "2 mg", "2 mg", "+2 mg", "2", "inf",
         "1e309 mg", "2e-1 mg"]
ROUTES = ["IV"] * 8 + ["SC", "PO", "intranasal", "PO/IV", "unknown", None] + ["IV"] * 5


def _definition(name):
    filename = "sofa2-dict.json" if name in {"delirium_tx_proxy", "other_vaso"} else "concept-dict.json"
    return json.loads((DATA / filename).read_text())[name]


def _projected_reader(data_source, monkeypatch):
    """Model parquet readers that return exactly the requested source columns."""
    load_table = data_source.load_table

    def load_projected(table_name, *args, **kwargs):
        table = load_table(table_name, *args, **kwargs)
        columns = kwargs.get("columns")
        if table_name == "medication" and columns is not None:
            keep = set(columns) | set(table.id_columns) | {table.index_column}
            table.data = table.data.loc[:, table.data.columns.isin(keep)]
        return table

    monkeypatch.setattr(data_source, "load_table", load_projected)


@pytest.mark.parametrize(("name", "database"), [
    (name, database) for name in DRUG_NAMES for database in ("eicu", "eicu_demo")
    if database in _definition(name)["sources"]
])
def test_rr_04_medication_exposure_requires_positive_dose_and_declared_route(
    tmp_path, monkeypatch, database, name,
):
    definition = _definition(name)
    assert definition["aggregate"] == "max"
    sources = [s for s in definition["sources"][database] if s.get("table") == "medication"]
    assert sources
    positive = {0, 6, 7, 14, 15, 18}
    if name not in IV_DRUGS:
        positive |= set(range(8, 14))
    for index, source_mapping in enumerate(sources):
        source = ConceptSource.from_mapping(source_mapping)
        is_code = source.sub_var == "drughiclseqno"
        frame = pd.DataFrame({
            "patientunitstayid": [1] * len(DOSES),
            "medicationid": range(len(DOSES)),
            "drugstartoffset": [60 * i for i in range(len(DOSES))],
            "drugname": ["" if is_code else DRUG_NAMES[name] + " 99 mg"] * len(DOSES),
            "drughiclseqno": [source.ids[0] if is_code else -1] * len(DOSES),
            "dosage": DOSES, "routeadmin": ROUTES,
            "drugordercancelled": ["No"] * len(DOSES),
        })
        allowed = pd.Series(True, index=frame.index)
        for column, pattern in source_mapping.get("exclude_regex", {}).items():
            allowed &= ~frame[column].astype("string").str.contains(pattern, case=False, na=False)
        eligible = set(frame.index[allowed])
        expected = positive & eligible
        direct = frame.copy()
        direct[name] = direct[source.value_var or "dosage"]
        out = _apply_callback(direct, source, name)
        assert set(out.index[out[name].eq(True)]) == expected  # noqa: E712
        assert not bool(out.loc[1, name])
        assert out.loc[[2, 3, 4, 5, 16, 17], name].isna().all()
        if name in IV_DRUGS:
            assert out.loc[[12, 13], name].isna().all()
            audit = out.attrs["eicu_medication_audit"]
            assert audit["route_unknown"] == 2
            assert audit["noniv_dropped"] == len({8, 9, 10, 11} & eligible)

        # Load actual parquet columns through the shipped source declaration,
        # including both the drugname and HICL fallbacks and the demo alias.
        data_source = _eicu_source(tmp_path / str(index), database, medication=frame)
        _projected_reader(data_source, monkeypatch)
        isolated = deepcopy(definition)
        isolated["sources"] = {database: [source_mapping]}
        dictionary = ConceptDictionary.from_payload({name: isolated})
        loaded = ConceptResolver(dictionary).load_concepts(
            [name], data_source, merge=False, r_compatible=False,
            interval=pd.Timedelta(hours=1), patient_ids={"patientunitstayid": [1]},
            concept_workers=1, verbose=False,
        )[name]
        result = loaded.data
        hours = result[loaded.index_column]
        if pd.api.types.is_timedelta64_dtype(hours):
            hours = hours / pd.Timedelta(hours=1)
        assert set(hours[result[name].eq(True)].astype(int)) == expected  # noqa: E712
        assert not result.loc[hours.eq(1), name].any()


def _fake_source(database, **tables):
    def load_table(name, **kwargs):
        if name not in tables:
            raise MissingTableSourceError(name)  # the source's absence signal
        return SimpleNamespace(data=tables[name].copy())
    return SimpleNamespace(config=SimpleNamespace(name=database), load_table=load_table)


@pytest.mark.parametrize("database", ["aumc", "sic", "eicu", "eicu_demo", "miiv", "mimic", "mimic_demo"])
@pytest.mark.parametrize("name", ["po2", "pco2", "ph", "sao2"])
def test_rr_01_unknown_specimens_never_enter_arterial_sources(database, name, caplog):
    """Unpaired, empty, unknown and conflicting labels cannot establish arterial blood."""
    from easyicu.concept.blood_gas_specimen import AUDIT_ATTR

    n = 8
    frame = pd.DataFrame({name: [90.] * n, "unit": "mmHg"})
    if database.startswith("eicu"):
        caplog.set_level("INFO", logger="easyicu.concept.eicu_source_qc")
        frame["patientunitstayid"] = 1
        frame["labresultoffset"] = range(n)
        samples = pd.DataFrame({
            "patientunitstayid": [1] * 9 + [2],
            "labotheroffset": [0, 1, 2, 3, 4, 5, 5, 6, 6, 7],
            "labothername": "Sample Type",
            "labothervaluetext": ["Arterial", "Venous", "", "Unspecified", "PA/pulmonary artery",
                                  "Arterial", "Unknown", "Mixed venous", "Arterial", "Arterial"],
            "labotherresult": "",
        })
        ds = _fake_source(database, customlab=samples)
    elif database == "sic":
        frame["CaseID"] = [1] * 4 + [2] * 4
        frame["LaboratoryType"] = [2296, 2297, None, "", None, "unknown", 2297, None]
        ds = _fake_source(database, cases=pd.DataFrame({"CaseID": [1, 2], "AdmissionYear": [2013, 2018]}))
    else:
        aumc = database == "aumc"
        token = "ART." if database in {"aumc", "miiv"} else "ART"
        keys = ["admissionid", "measuredat"] if aumc else ["subject_id", "charttime"]
        frame[keys[0]] = 1
        frame[keys[1]] = pd.date_range("2150-01-01", periods=n, freq="h")
        frame["specimen_id"] = range(n)
        specimens = frame.iloc[[0, 1, 2, 3, 4, 5, 5, 6]].copy()
        specimens["value"] = [token, "VEN.", "", "UNKNOWN", "PAP", token, "UNKNOWN", "MIX"]
        specimens["itemid"] = 11646 if aumc else 52033 if database == "miiv" else 50800
        ds = _fake_source(database, **{
            "freetextitems" if aumc else "labevents": specimens,
            "listitems": pd.DataFrame(columns=[*keys, "itemid", "value"]),
        })
    table = {"aumc": "numericitems", "sic": "laboratory"}.get(database, "lab" if database.startswith("eicu") else "labevents")
    for mapping in _definition(name)["sources"][database]:
        if mapping["table"] != table:
            continue
        source = ConceptSource.from_mapping(mapping)
        out = _apply_callback(frame, source, name, unit_column="unit", data_source=ds)
        if database.startswith("eicu"):
            # R6 refined: eICU has no per-sample specimen field beyond these
            # few Sample Type rows. paO2/paCO2 are arterial by name; pH and
            # SaO2 need a same-offset paO2/paCO2, absent from this source.
            audit = out.attrs["eicu_specimen_audit"]
            # Round 8 (R7-F018): a recorded blank/Unspecified sample is not
            # "no sample"; the label and same-panel fallbacks do not apply.
            if name in {"po2", "pco2"}:
                assert out.index.tolist() == [0, 5, 7]
                assert audit == {"explicit_arterial": 2, "label_defined_arterial": 1,
                                 "co_measured_arterial": 0, "excluded_venous": 3,
                                 "excluded_recorded_unknown": 2, "excluded_unknown": 0}
            else:
                assert out.index.tolist() == [0, 5]
                assert audit == {"explicit_arterial": 2, "label_defined_arterial": 0,
                                 "co_measured_arterial": 0, "excluded_venous": 3,
                                 "excluded_recorded_unknown": 2, "excluded_unknown": 1}
            assert any(record.args == audit for record in caplog.records)
            continue
        assert out.index.tolist() == [0]
        audit = out.attrs[AUDIT_ATTR]
        assert audit["unrecorded_rows_dropped"] > 0
        assert audit["unrecorded_rows_kept"] == 0
    if database.startswith("eicu"):
        # A missing optional specimen table supplies no arterial evidence.
        missing = _apply_callback(frame, source, name, data_source=_fake_source(database))
        assert len(missing) == (n if name in {"po2", "pco2"} else 0)
        assert any(record.args == missing.attrs["eicu_specimen_audit"] for record in caplog.records)


@pytest.mark.parametrize("database", ["eicu", "eicu_demo"])
def test_rr_02_lab_saturation_uses_same_arterial_evidence_while_pulse_survives(tmp_path, database):
    from tests.support.eicu_source import _hourly

    ds = _eicu_source(tmp_path, database,
        lab=pd.DataFrame({"labname": "O2 Sat (%)", "labresult": [98., 60., 70.],
                          "labresultoffset": [0, 60, 120], "labmeasurenameinterface": "%"}),
        customlab=pd.DataFrame({"labothername": "Sample Type", "labotheroffset": [0, 60],
                                "labothervaluetext": ["Arterial", "Venous"], "labotherresult": ""}),
        vitalperiodic=pd.DataFrame({"observationoffset": [180], "sao2": [95.]}))
    assert _hourly(ds, "o2sat") == {0.: 98., 3.: 95.}
    assert _hourly(ds, "sao2") == {0.: 98.}


@pytest.mark.parametrize("database", ["eicu", "eicu_demo", "sic"])
@pytest.mark.parametrize("name", ["be", "tco2"])
def test_rr_03_base_excess_and_total_co2_require_arterial_specimen(tmp_path, database, name):
    if database == "sic":
        from tests.support.blood_gas_sources import _source, _write, hourly
        rows = pd.DataFrame({"CaseID": 1, "id": [1, 2, 3], "Offset": [7500, 11100, 14700],
                             "LaboratoryID": 668 if name == "be" else 670,
                             "LaboratoryType": [2296, 2297, None],
                             "LaboratoryValue": [2., 3., 4.] if name == "be" else [22., 23., 24.]})
        _write(rows, tmp_path / "laboratory/part.parquet")
        _write(pd.DataFrame({"CaseID": [1], "PatientID": [11], "AdmissionYear": [2013],
                             "ICUOffset": [0], "TimeOfStay": [172800]}), tmp_path / "cases.parquet")
        got = hourly(_source(tmp_path, "sic"), "CaseID", name)
        assert got == {2.: 2. if name == "be" else 22.}
    else:
        from tests.support.eicu_source import _hourly
        labels = ["Base Excess", "Base Deficit"] if name == "be" else ["Total CO2"]
        lab = pd.DataFrame([{"labname": label, "labresult": value, "labresultoffset": i * 60 + j * 180,
                             "labmeasurenameinterface": "mmol/L"}
                            for j, label in enumerate(labels)
                            for i, value in enumerate([2., 3., 4.] if name == "be" else [22., 23., 24.])])
        sample = pd.DataFrame({"labothername": "Sample Type", "labotheroffset": [0, 60, 180, 240],
                               "labothervaluetext": ["Arterial", "Venous"] * 2, "labotherresult": ""})
        ds = _eicu_source(tmp_path, database, lab=lab, customlab=sample)
        # R6 rule 4: eICU Total CO2 is the chemistry panel, not an arterial tco2.
        assert _hourly(ds, name) == ({0.: 2., 3.: -2.} if name == "be" else {})


AUMC_IV_RESIDUES = "albumin_iv bicarbonate fentanyl heparin magnesium_iv mannitol milrinone nicardipine nitroglycerin potassium_iv propofol rocuronium".split()
AUMC_IV_GUARDS = "amiodarone calcium_iv levetiracetam midazolam morphine pantoprazole".split()
AUMC_INFUSIONS = "norepi_rate epi_rate dopa_rate dobu_rate phn_rate propofol_rate fentanyl_rate midazolam_rate dopa_dur dobu_dur epi_dur norepi_dur".split()


@pytest.mark.parametrize("name", AUMC_IV_RESIDUES + AUMC_IV_GUARDS + AUMC_INFUSIONS + ["dex", "other_vaso"])
def test_rr_05_iv_drug_orders_exclude_niet_iv_categories_on_raw_loading(tmp_path, name):
    from tests.support.aumc_source import aumc_db as _db, aumc_drug as _drug, aumc_load as _load

    for source in _definition(name)["sources"]["aumc"]:
        item = source["ids"] if isinstance(source["ids"], int) else source["ids"][0]
        unit = "ml" if name == "dex" else "mg" if name == "midazolam_rate" else "mcg"
        rate = "uur" if name in {"dex", "fentanyl_rate", "midazolam_rate"} else "min"
        db = _db(tmp_path, drugs=[_drug(5 + i * 60, 10 + i * 60, item, unit=unit, rate=rate, category=category)
                                  for i, category in enumerate(["IV", "Niet iv Tabletten", " niet iv Overig", "NIET IV Circulatie/Diurese"])])
        if name in {"fentanyl_rate", "midazolam_rate"}:
            # These mass-rate fixtures have no per-kg dose or weight dependency.
            path = tmp_path / "drugitems.parquet"
            raw = pd.read_parquet(path).assign(doserateperkg=False)
            raw.to_parquet(path, index=False)
        got = _load(db, name, sofa=name == "other_vaso")
        assert set(got) == {5 / 60 if name.endswith("_dur") else 0.}
        assert all(value > 0 for value in got.values())
        if name in AUMC_IV_RESIDUES + AUMC_IV_GUARDS:
            assert got == {0.: True}
        assert "ordercategory" in source.get("extra_vars", [])


HIRID_IV = "amiodarone bicarbonate calcium_iv dexmedetomidine dextrose50 fentanyl heparin levetiracetam magnesium_iv midazolam milrinone morphine nitroglycerin potassium_iv propofol rocuronium other_vaso".split()
HIRID_INFUSIONS = "adh_rate dobu_dur dobu_rate epi_dur epi_rate fentanyl_rate midazolam_rate norepi_dur norepi_rate propofol_rate".split()


@pytest.mark.parametrize("name", HIRID_IV + HIRID_INFUSIONS)
def test_rr_06_iv_route_filter_preserves_separate_infusion_groups(tmp_path, name):
    from tests.support.hirid_source import hirid_load as _load, hirid_source_root as _source_root

    source = _definition(name)["sources"]["hirid"][0]
    assert "route" in source.get("extra_vars", [])
    item = source["ids"] if isinstance(source["ids"], int) else source["ids"][0]
    unit = "U" if name == "adh_rate" else "mg" if name == "midazolam_rate" else "µg"
    kwargs = {"dict_path": DATA / "sofa2-dict.json"} if name == "other_vaso" else {}
    rows = [{"patientid": 1, "givenat": t, "pharmaid": item, "givendose": dose,
             "doseunit": unit, "route": route, "infusionid": infusion, "recordstatus": 0}
            for infusion, route, amount in [(7, "iv-inf", 12.), (8, "cv-inf", 24.), (9, "p os", 900.),
                                             (10, "ep-inj", 900.), (11, "sc", 900.), (12, "inh", 900.), (13, None, 900.)]
            for t, dose in [(.25, 0.), (1.25, amount), (2.25, amount)]]
    root = _source_root(tmp_path, "pharma", rows)
    out = _load(root, [name], **kwargs)[name]
    assert out[name].notna().any()
    path = root / "pharma/1.parquet"
    raw = pd.read_parquet(path)
    raw.loc[raw.route.isin(["iv-inf", "cv-inf"])].to_parquet(path, index=False)
    pd.testing.assert_frame_equal(out, _load(root, [name], **kwargs)[name])
    # The filter must also run before a grouped rate/duration callback consumes columns.
    direct = _apply_callback(pd.DataFrame(rows).rename(columns={"givendose": name}),
                             ConceptSource.from_mapping(source), name,
                             data_source=_fake_source("hirid"))
    if "infusionid" in direct:
        assert set(direct.infusionid) == {7, 8}


@pytest.mark.parametrize("layout", ["directory", "file"])  # full MIMIC-IV / public demo conversion
@pytest.mark.parametrize("drug,item,dose", [("furosemide", 221794, 40.), ("bumetanide", 229639, 1.)])
def test_rr_07_emar_dedup_matches_real_drug_dose_time_and_preserves_oral(tmp_path, drug, item, dose, layout):
    from tests.support.mimic_synthetic_source import mimic_synthetic_source as _synthetic_source, mimic_synthetic_load as _load

    t = pd.Timestamp("2150-01-01")
    hours = [0., 3., 6., 9., 12., 15., 18., 21., 24., 27., 30., 33.]
    routes = ["IV", "PO"] + ["IV"] * 9 + [None]
    n = len(hours)
    emar = pd.DataFrame({"subject_id": 11, "hadm_id": 21, "emar_id": [f"event-{i}" for i in range(n)],
                         "pharmacy_id": range(n), "medication": drug,
                         "charttime": t + pd.to_timedelta(hours, unit="h"), "event_txt": "Administered"})
    details = pd.DataFrame({"emar_id": emar.emar_id, "parent_field_ordinal": 1.,
                            "dose_given": str(dose), "dose_given_unit": "mg", "route": routes})
    inputs = pd.DataFrame({"subject_id": 11, "hadm_id": [21] * 8 + [22] + [21] * 3,
                           "stay_id": [1] * 8 + [2] + [1] * 3, "orderid": range(n), "linkorderid": range(n),
                           "itemid": [item] * 4 + [229639 if drug == "furosemide" else 221794] + [item] * 7,
                           "starttime": t + pd.to_timedelta([.5, 3., 6., 10.01, 12., 15., 18., 21., 24., 28., 29., 33.], unit="h"),
                           "amount": [dose / 1000, dose, dose * 2, dose, dose, 0., dose, dose] + [dose] * 4,
                           "amountuom": ["g"] + ["mg"] * 5 + ["mL"] + ["mg"] * 5,
                           "statusdescription": ["FinishedRunning"] * 7 + ["Rewritten"] + ["FinishedRunning"] * 4,
                           "rate": 0., "rateuom": "mg/hour"})
    inputs["endtime"] = inputs.starttime + pd.Timedelta(minutes=1)
    orders = pd.DataFrame({"pharmacy_id": range(n), "route": routes})
    ds = _synthetic_source(tmp_path, "miiv", emar=emar, emar_detail=details, prescriptions=orders, inputevents=inputs)
    for table in ("emar_detail", "prescriptions") if layout == "directory" else ():
        directory = tmp_path / table
        directory.mkdir()
        (tmp_path / f"{table}.parquet").rename(directory / "part.parquet")
    # Isolate the actual eMAR source; the surviving ICU input copy is published by its own source.
    mapping = deepcopy(_definition("loop_diuretic_furosemide_eq"))
    mapping["sources"] = {"miiv": [s for s in mapping["sources"]["miiv"] if s["table"] == "emar" and s["fe_drug"] == drug]}
    loaded = ConceptResolver(ConceptDictionary.from_payload({"loop_diuretic_furosemide_eq": mapping})).load_concepts(
        ["loop_diuretic_furosemide_eq"], ds, merge=False, r_compatible=False,
        interval=pd.Timedelta(hours=1), patient_ids={"stay_id": [1]}, concept_workers=1, verbose=False,
    )["loop_diuretic_furosemide_eq"]
    result = loaded.data.set_index(loaded.index_column).loop_diuretic_furosemide_eq.to_dict()
    factor = 1. if drug == "furosemide" else 40.
    kept = hours[1:9] + ([33.] if drug == "bumetanide" else [])
    assert result == {h: dose * factor * (.5 if drug == "furosemide" and h == 3. else 1.) for h in kept}
    # The complete concept counts the matched pair once, while retaining the oral dose.
    complete = _load(ds, ["loop_diuretic_furosemide_eq"])["loop_diuretic_furosemide_eq"].data
    clock = next(c for c in ("starttime", "charttime", "time") if c in complete)
    assert complete.loc[complete[clock].eq(0), "loop_diuretic_furosemide_eq"].sum() == pytest.approx(dose * factor)
    # Nullable converted identifiers may be floats in eMAR and integers in inputs.
    from easyicu.concept.emar_diuretics import drop_inputevent_duplicates

    for dtype in ("float64", "Float64", "Int64"):
        normalized = pd.DataFrame({"hadm_id": pd.Series([21, None], dtype=dtype),
                                   "charttime": t, "_route": "iv", "_dose": dose, "_unit": "mg"})
        kept = drop_inputevent_duplicates(normalized, inputs.iloc[[0]], key="hadm_id", time_col="charttime", drug=drug)
        assert kept.index.tolist() == [1]
    # Verify the helper boundary too: the caller's SQL filters are not the only guard.
    for change in ({"itemid": 229639 if drug == "furosemide" else 221794},
                   {"hadm_id": 22}, {"amount": dose * 2, "amountuom": "mg"},
                   {"starttime": t + pd.Timedelta(minutes=61)},
                   {"amount": dose, "amountuom": "mL"}, {"statusdescription": "Rewritten"}):
        different = inputs.iloc[[0]].assign(**change)
        kept = drop_inputevent_duplicates(normalized, different, key="hadm_id", time_col="charttime", drug=drug)
        assert kept.index.tolist() == [0, 1]


@pytest.mark.parametrize("name", ["map", "map_invasive"])
def test_rr_08_secondary_map_follows_primary_then_secondary_then_cuff(tmp_path, name):
    from tests.support.aumc_source import aumc_db as _db, aumc_load as _load

    rows = [(5, 6642, 100., "mmHg"), (5, 8843, 80., "mmHg"), (5, 6679, 90., "mmHg"),
            (65, 8843, 80., "mmHg"), (65, 6679, 90., "mmHg"),
            (125, 8843, 0., "mmHg"), (125, 6679, 90., "mmHg")]
    expected = {0.: 100., 1.: 80., 2.: 90.} if name == "map" else {0.: 100., 1.: 80.}
    assert _load(_db(tmp_path, numeric=rows), name) == expected


@pytest.mark.parametrize("text,expected", [
    ("81 3", True), ("12.5 3", True), ("0 3", False),     # numeric unit codes
    ("1,000 mg", True), ("1,000", True), ("1,00 mg", None),  # thousands separators
    ("2-4 mg", True), ("2 - 4 mg", True), ("2.5-5 mg", True),
    ("0-2 mg", None), ("4-2 mg", None), ("0,5 mg", None),   # sliding scale, reversed, decimal comma
    ("325 mg (1 tab)", True), ("100mg", True), ("held", None),
])
def test_rr_04_dose_text_reads_unit_codes_separators_and_ranges(text, expected):
    from easyicu.concept.eicu_medication_evidence import _positive_dose

    got = _positive_dose(pd.Series([text], dtype=object)).iloc[0]
    assert (None if pd.isna(got) else bool(got)) is expected


@pytest.mark.parametrize("database", ["eicu", "eicu_demo"])
@pytest.mark.parametrize("name", ["ph", "sao2", "be"])
def test_r6_eicu_ambiguous_gases_need_a_same_offset_arterial_anchor(database, name):
    """pH, SaO2 and BE enter beside a paO2/paCO2 at the same stay and offset."""
    frame = pd.DataFrame({"patientunitstayid": [1, 1, 1, 1, 2], "labresultoffset": [0, 60, 120, 180, 0],
                          name: [7.4] * 5, "unit": ""})
    lab = pd.DataFrame({  # anchors: 0 paO2, 60 paCO2, 120 a blank paO2, 180 venous, stay 2 none
        "patientunitstayid": [1, 1, 1, 1, 1], "labresultoffset": [0, 60, 120, 180, 181],
        "labname": ["paO2", "paCO2", "paO2", "paO2", "paO2"], "labresult": [80., 40., None, 40., 80.],
    })
    samples = pd.DataFrame({"patientunitstayid": [1], "labotheroffset": [180], "labothername": "Sample Type",
                            "labothervaluetext": ["Venous"], "labotherresult": ""})
    mapping = next(m for m in _definition(name)["sources"][database] if m["table"] == "lab")
    out = _apply_callback(frame, ConceptSource.from_mapping(mapping), name,
                          data_source=_fake_source(database, lab=lab, customlab=samples))
    assert out.labresultoffset.tolist() == [0, 60]
    assert set(out.specimen_status) == {"co_measured_arterial"}
    assert out.attrs["eicu_specimen_audit"] == {
        "explicit_arterial": 0, "label_defined_arterial": 0, "co_measured_arterial": 2,
        "excluded_venous": 1, "excluded_recorded_unknown": 0, "excluded_unknown": 2}


def test_r6_eicu_total_co2_is_not_an_arterial_tco2_source():
    for database in ("eicu", "eicu_demo"):
        assert database not in _definition("tco2")["sources"]
