"""Round-6 extraction regressions; all source records are invented."""

import json
import os
from pathlib import Path
import tempfile

import pandas as pd
import pytest

import easyicu
from easyicu.concept import ConceptDictionary, ConceptResolver
from easyicu.concept.metadata_projection import (
    ColumnProjectionSpec,
    ConceptColumnRole,
    project_concept_column_metadata,
)
from easyicu.utils.callback_utils import hirid_rate, hirid_rate_kg, hirid_rate_mass
from tests.support.blood_gas_sources import _source, _write
from tests.support.eicu_source import _eicu_source

DICTIONARY = Path(easyicu.__file__).parent / "data/concept-dict.json"
MBAR_TO_CMH2O = 100 / 98.0665
AUMC_ITEMS = {
    "peep": [12284, 12336, 8862, 16250],
    "peep_set": [12284, 12336, 8862, 16250],
    "peep_total": [12301, 12364],
    "pip": [12281, 8877, 6698, 12365, 16239],
    "plateau_pres_raw": [12280, 8878],
    "plateau_pres": [12280, 8878],
    "mean_airway_pres": [12278, 12362],
    "ps": [12286, 12338, 16251, 8865],
    "compliance": [6710, 12561],
}


def _load(source, names, id_column):
    return ConceptResolver(ConceptDictionary.from_json(DICTIONARY)).load_concepts(
        names, source, merge=False, r_compatible=False,
        interval=pd.Timedelta(hours=1), patient_ids={id_column: [1]},
        concept_workers=1, verbose=False,
    )


def _values(table, name):
    frame = table.data.dropna(subset=[name])
    return frame.set_index(table.index_column)[name].to_dict()


def _assert_native_columns(database, table, frame):
    configs = json.loads((DICTIONARY.parent / "data-sources.json").read_text())
    config = next(c for c in configs if c["name"] == database)
    assert set(frame) <= set(config["tables"][table]["cols"])


def _published(tmp_path, monkeypatch, database, module, id_column, extra_modules=()):
    from easyicu.api.extraction import extract_database

    monkeypatch.setattr(tempfile, "tempdir", tempfile.tempdir)
    for variable in ("TMPDIR", "EASYICU_DUCKDB_TEMP_DIR"):
        if variable in os.environ:
            monkeypatch.setenv(variable, os.environ[variable])
        else:
            monkeypatch.delenv(variable, raising=False)
    output = tmp_path / "published"
    result = extract_database(
        database, data_path=str(tmp_path), output_dir=str(output),
        modules=["demographics", "outcome", module, *extra_modules], patient_ids={id_column: [1]},
        native_export_v2=True, resource_budget_mb=512, verbose=False,
    )
    assert not result["modules"][module].get("errors")
    return pd.read_parquet(output / f"{module}.parquet")


def _published_values(frame, name):
    assert frame.stay_id.eq(1).all()
    return frame.dropna(subset=[name]).set_index("charttime")[name].to_dict()


@pytest.mark.parametrize("layout", ["flat", "bucket"])
@pytest.mark.parametrize("name", AUMC_ITEMS)
def test_r6a_01_aumc_converts_each_native_row_before_channel_selection_and_hourly_median(
    tmp_path, monkeypatch, name, layout,
):
    compliance = name == "compliance"
    native_unit, canonical_unit = ("ml/mbar", "mL/cmH2O") if compliance else ("mbar", "cmH2O")
    factor = 1 / MBAR_TO_CMH2O if compliance else MBAR_TO_CMH2O
    items = AUMC_ITEMS[name]
    # Conversion changes the ordering of the first two values. The median of
    # the converted rows differs from converting a raw median (or using min).
    # Unequal counts per item also distinguish a pooled median from a median
    # of item medians. Total PEEP keeps these rows in its primary channel.
    canonical_values = [19.9, 10.] if compliance else [20.1, 30.]
    canonical_item = items[0] if name == "peep_total" else items[1]
    rows = [(15., items[0], 20., native_unit),
            (30., canonical_item, canonical_values[0], canonical_unit),
            (45., canonical_item, canonical_values[1], canonical_unit)]
    expected = {0.: 20. * factor}
    # Every declared item must reach the public selector, with its own unit.
    for hour, item in enumerate(items, start=1):
        rows.append((hour * 60. + 5., item, 12., native_unit))
        expected[float(hour)] = 12. * factor
    if name == "peep":
        # Settings win over measured PEEP, which only fills absent settings.
        rows.extend([(10., 8879, 35., native_unit), (1205., 8879, 8., native_unit)])
        expected[20.] = 8. * factor
    if name == "peep_total":
        # The primary total-PEEP channel wins; the secondary remains a fallback.
        rows.append((10., 12364, 35., native_unit))
    # Excluded alarm, measured, and pressure-control channels cannot alter the
    # requested quantity or fabricate an hour when that quantity is absent.
    for item in {"peep_set": [8879, 6694, 12354, 9666], "pip": [8852],
                 "ps": [12285], "peep_total": [8879, 12284]}.get(name, []):
        rows.extend([(10., item, 39., native_unit), (1265., item, 39., native_unit)])
    bound = {"peep": 40., "peep_set": 40., "peep_total": 40., "pip": 100.,
             "plateau_pres_raw": 100., "plateau_pres": 100.,
             "mean_airway_pres": 60., "ps": 40., "compliance": 200.}[name]
    # Apply bounds to canonical values, so a native in-range pressure that
    # converts above the cap drops, and a native compliance above it can survive.
    rows.append((1325., items[0], bound * (1.01 if compliance else .99), native_unit))
    if compliance:
        expected[22.] = bound * 1.01 * factor
    if name == "plateau_pres":
        rows.extend([(1805., 12280, 20., "mbar"), (1805., 6698, 19.2, "cmH2O"),
                     (1865., 12280, 20., "cmH2O"), (1865., 12284, 20., "mbar")])
        # Converted plateau > PIP + 1 or < converted PEEP is inconsistent.

    origin = 17.25  # Deliberately off the hour grid.
    _write(pd.DataFrame({
        "admissionid": [1], "patientid": [11], "admittedat": [origin * 60_000],
        "dischargedat": [(origin + 48 * 60) * 60_000],
        "dateofdeath": [float("nan")], "destination": ["Home"],
    }), tmp_path / "admissions.parquet")
    frame = pd.DataFrame(rows, columns=["measuredat", "itemid", "value", "unit"])
    frame["measuredat"] = (frame["measuredat"] + origin) * 60_000
    frame["registeredat"] = frame["measuredat"]
    frame["admissionid"] = 1
    frame["tag"] = None
    frame["unitid"] = 999  # The row's text unit, not a guessed item/unit ID, governs conversion.
    _assert_native_columns("aumc", "numericitems", frame)
    path = tmp_path / ("numericitems.parquet" if layout == "flat" else "numericitems_bucket/bucket_id=0/part.parquet")
    _write(frame, path)
    source = _source(tmp_path, "aumc")
    # Include sibling concepts to exercise the shared-table/batch loader too.
    names = list(AUMC_ITEMS)
    table = _load(source, names, "admissionid")[name]
    assert _values(table, name) == pytest.approx(expected)
    # Hourly tables intentionally drop row unit columns; export labels come
    # from the dictionary through the public metadata projection.
    metadata = project_concept_column_metadata(
        ConceptDictionary.from_json(DICTIONARY).get(name),
        spec=ColumnProjectionSpec(name, name, ConceptColumnRole.VALUE),
        source_database="aumc",
    )
    assert metadata.canonical_unit == canonical_unit
    single = _load(source, [name], "admissionid")[name]
    assert _values(single, name) == pytest.approx(expected)
    if name in {"peep", "pip"}:
        # The audit concerns published hours too: exercise native publication,
        # including its channel choice, canonical clock and aggregation.
        published = _published(tmp_path, monkeypatch, "aumc", "ventilator", "admissionid")
        got = _published_values(published, name)
        assert got == pytest.approx(expected)


@pytest.mark.parametrize("name,unit,target", [
    ("adh_rate", "U", None),
    ("fentanyl_rate", "mg", "mcg/hour"),
    ("midazolam_rate", "µg", "mg/hour"),
    ("norepi_rate", "µg", "kg"),
    ("epi_rate", "µg", "kg"),
    ("dobu_rate", "µg", "kg"),
    ("propofol_rate", "µg", "kg"),
])
def test_r6a_02_hirid_duplicate_times_are_permutation_invariant_and_conserve_dose(name, unit, target):
    frame = pd.DataFrame({
        "patientid": [1] * 9 + [2] * 3,
        "infusionid": [7] * 6 + [8] * 3 + [7] * 3,
        "givenat": [.75, 1.25, 1.25, 2., 10., 1., .5, 1.25, 1.25, .9, 1.1, 1.1],
        "givendose": [0., 6., 18., 6., 3., 999., 0., 4., 8., 0., 2., 6.],
        "recordstatus": [0] * 5 + [2] + [0] * 6,
        "doseunit": unit, "weight": [80.] * 9 + [40.] * 3,
    })
    if target:
        # Concurrent doses can use different compatible mass units. Convert
        # those rows before summing, without giving either row a zero interval.
        frame.loc[2, "doseunit"] = "µg" if unit == "mg" else "mg"
        frame.loc[2, "givendose"] *= 1000. if unit == "mg" else .001
    kwargs = dict(concept_name=name, val_col="givendose", unit_col="doseunit",
                  grp_var="infusionid", index_col="givenat")

    def run(rows):
        if target == "kg":
            out = hirid_rate_kg(rows, **kwargs)
        elif target:
            out = hirid_rate_mass(rows, target_unit=target, **kwargs)
        else:
            out = hirid_rate(rows, **kwargs)
        return out.sort_values(["patientid", "givenat"]).reset_index(drop=True)

    baseline = run(frame)
    for permuted in (frame.iloc[::-1], frame.sample(frac=1, random_state=42),
                     frame.sample(frac=1, random_state=73).reset_index(drop=True)):
        pd.testing.assert_frame_equal(run(permuted), baseline)
    raw = {(1, 0.): 20., (1, 1.): 22., (1, 10.): 3., (2, 0.): 4., (2, 1.): 4.}
    scale = 1000. if name == "fentanyl_rate" else .001 if name == "midazolam_rate" else 1 / 60
    expected = {key: dose / (60 * (80 if key[0] == 1 else 40)) if target == "kg" else dose * scale
                for key, dose in raw.items()}
    assert baseline.set_index(["patientid", "givenat"])[name].to_dict() == pytest.approx(expected)


def test_r6a_01_dense_canonical_pressure_keeps_raw_median_on_the_icu_clock(tmp_path, monkeypatch):
    # Just cross the 1000-row presampling branch with invented records. Every
    # value already uses cmH2O: this isolates an aggregation/clock error from
    # the 2% unit factor. Absolute-hour presampling splits one ICU hour in two.
    origin = 17.25 * 60_000
    _write(pd.DataFrame({"admissionid": [1], "patientid": [11], "admittedat": [origin],
                         "dischargedat": [origin + 12 * 3_600_000],
                         "dateofdeath": [float("nan")], "destination": ["Home"]}),
           tmp_path / "admissions.parquet")
    frame = pd.DataFrame({"measuredat": [15., 30., 45.] * 334, "value": [10., 20., 35.] * 334,
                          "unit": "cmH2O", "itemid": 12281, "admissionid": 1, "tag": None})
    frame["measuredat"] = origin + frame["measuredat"] * 60_000
    frame["registeredat"] = frame["measuredat"]
    _assert_native_columns("aumc", "numericitems", frame)
    _write(frame, tmp_path / "numericitems.parquet")
    source = _source(tmp_path, "aumc")
    for names in (["pip"], ["pip", "peep"]):
        assert _values(_load(source, names, "admissionid")["pip"], "pip") == {0.: 20.}
    published = _published(tmp_path, monkeypatch, "aumc", "ventilator", "admissionid")
    assert _published_values(published, "pip") == {0.: 20.}


@pytest.mark.parametrize("label", ["Mechanical Ventilator Mode", "Ventilator Support Mode"])
@pytest.mark.parametrize("layout", ["flat", "bucket"])
def test_r6a_03_eicu_mode_labels_reach_packaged_public_selector(tmp_path, monkeypatch, label, layout):
    rows = pd.DataFrame({
        "respchartoffset": [5, 65, 125, 185, 245],
        "respchartvaluelabel": [label] * 4 + ["Unrelated Mode"],
        "respchartvalue": ["APVcmv", "SIMV", "Spontaneous", "AC/CMV", "APVcmv"],
    })
    _assert_native_columns("eicu", "respiratorycharting", rows)
    source = _eicu_source(tmp_path, "eicu", respiratorycharting=rows)
    patient = pd.read_parquet(tmp_path / "patient.parquet").assign(admissionheight=175.)
    _assert_native_columns("eicu", "patient", patient)
    _write(patient, tmp_path / "patient.parquet")
    if layout == "bucket":
        path = tmp_path / "respiratorycharting_bucket/bucket_id=0/part.parquet"
        path.parent.mkdir(parents=True)
        (tmp_path / "respiratorycharting.parquet").rename(path)
    names = ["vent_mode", "vent_breath_seq"]
    tables = _load(source, names, "patientunitstayid")
    assert _values(tables["vent_mode"], "vent_mode") == {0.: "dual_adaptive"}
    assert _values(tables["vent_breath_seq"], "vent_breath_seq") == {1.: "simv", 2.: "spontaneous"}
    for name in names:
        assert _values(_load(source, [name], "patientunitstayid")[name], name) == _values(tables[name], name)
    published = _published(tmp_path, monkeypatch, "eicu", "ventilator", "patientunitstayid")
    for name in names:
        assert _published_values(published, name) == _values(tables[name], name)


@pytest.mark.parametrize("layout", ["flat", "bucket"])
def test_r6a_02_hirid_public_rates_preserve_duplicate_doses_in_every_path(tmp_path, monkeypatch, layout):
    names = ["adh_rate", "fentanyl_rate", "midazolam_rate", "norepi_rate",
             "epi_rate", "dobu_rate", "propofol_rate"]
    definitions = json.loads(DICTIONARY.read_text())
    origin = pd.Timestamp("2180-01-01 00:17:15")
    general = pd.DataFrame({"patientid": [1], "admissiontime": [origin], "age": [60],
                            "sex": ["M"], "discharge_status": ["alive"]})
    observations = pd.DataFrame({"patientid": [1, 1], "datetime": [origin, origin + pd.Timedelta(hours=12)],
                                 "variableid": [10000400, 200], "value": [80., 75.], "status": [0, 0]})
    _assert_native_columns("hirid", "general", general)
    _assert_native_columns("hirid", "observations", observations)
    _write(general, tmp_path / "general.parquet")
    _write(observations, tmp_path / "observations/part.parquet")
    rows, expected = [], {}
    for infusion, name in enumerate(names, start=1):
        ids = definitions[name]["sources"]["hirid"][0]["ids"]
        item = ids[0] if isinstance(ids, list) else ids
        unit = "U" if name == "adh_rate" else "µg"
        # All rates stay within declared bounds. Weight comes from the native
        # observations table, never from an invented pharma.weight column.
        for time, dose in [(.75, 0.), (1.25, .6), (1.25, 1.8), (2., .6)]:
            rows.append({"patientid": 1, "pharmaid": item, "infusionid": infusion,
                         "givenat": origin + pd.Timedelta(hours=time), "givendose": dose,
                         "doseunit": unit, "recordstatus": 0, "route": "cv-inf"})
        scale = {"adh_rate": 1 / 60, "fentanyl_rate": 1., "midazolam_rate": .001}.get(name, 1 / (60 * 80))
        expected[name] = {0.: 1.2 * scale, 1.: 1.8 * scale}
    frame = pd.DataFrame(rows)
    _assert_native_columns("hirid", "pharma", frame)
    path = tmp_path / ("pharma/part.parquet" if layout == "flat" else "pharma_bucket/bucket_id=0/part.parquet")
    for rows in (frame, frame.iloc[::-1], frame.sample(frac=1, random_state=42)):
        _write(rows, path)
        source = _source(tmp_path, "hirid")
        tables = _load(source, names, "patientid")
        for name in names:
            assert _values(tables[name], name) == pytest.approx(expected[name])
            assert _values(_load(source, [name], "patientid")[name], name) == pytest.approx(expected[name])
    published = _published(tmp_path, monkeypatch, "hirid", "medications", "patientid", ("vasopressors",))
    vasopressors = pd.read_parquet(tmp_path / "published/vasopressors.parquet")
    for name in names:
        module = published if name in {"fentanyl_rate", "midazolam_rate", "propofol_rate"} else vasopressors
        assert _published_values(module, name) == pytest.approx(expected[name])


@pytest.mark.parametrize("layout", ["flat", "bucket"])
def test_r6_f020_aumc_measured_6694_fills_only_hours_without_valid_settings(tmp_path, monkeypatch, layout):
    origin = 17.25 * 60_000
    admissions = pd.DataFrame({"admissionid": [1], "patientid": [11], "admittedat": [origin],
                               "dischargedat": [origin + 12 * 3_600_000],
                               "dateofdeath": [float("nan")], "destination": ["Home"]})
    _assert_native_columns("aumc", "admissions", admissions)
    _write(admissions, tmp_path / "admissions.parquet")
    frame = pd.DataFrame([
        (5., 12284, 10., "mbar"), (10., 6694, 15., "mbar"), (15., 8879, 16., "mbar"),
        (65., 6694, 8., "mbar"), (95., 6694, 10., "cmH2O"),
        (125., 8879, 12., "cmH2O"),
        (185., 8862, 39.9, "mbar"), (190., 6694, 6., "mbar"),
        (245., 6694, 7., "cmH2O"), (250., 8879, 9., "mbar"),
        (305., 12354, 30., "mbar"), (310., 9666, 30., "mbar"),
        (365., 12301, 11., "mbar"), (425., 6694, None, "mbar"),
    ], columns=["measuredat", "itemid", "value", "unit"])
    frame["measuredat"] = origin + frame["measuredat"] * 60_000
    frame["registeredat"] = frame["measuredat"]
    frame["admissionid"] = 1
    frame["tag"] = None
    _assert_native_columns("aumc", "numericitems", frame)
    path = tmp_path / ("numericitems.parquet" if layout == "flat" else "numericitems_bucket/bucket_id=0/part.parquet")
    _write(frame, path)
    source = _source(tmp_path, "aumc")
    names = ["peep", "peep_set", "peep_total"]
    tables = _load(source, names, "admissionid")
    factor = MBAR_TO_CMH2O
    expected = {"peep": {0.: 10 * factor, 1.: (8 * factor + 10) / 2, 2.: 12.,
                          3.: 6 * factor, 4.: (7 + 9 * factor) / 2},
                "peep_set": {0.: 10 * factor}, "peep_total": {6.: 11 * factor}}
    for name in names:
        assert _values(tables[name], name) == pytest.approx(expected[name])
        assert _values(_load(source, [name], "admissionid")[name], name) == pytest.approx(expected[name])
    published = _published(tmp_path, monkeypatch, "aumc", "ventilator", "admissionid")
    for name in names:
        assert _published_values(published, name) == pytest.approx(expected[name])
    # Same synthetic batch before adding 6694: quantify net coverage separately
    # from recomputing hours where both measured channels now contribute.
    _write(frame.loc[frame.itemid.ne(6694)], path)
    previous = _values(_load(_source(tmp_path, "aumc"), ["peep"], "admissionid")["peep"], "peep")
    current = _values(tables["peep"], "peep")
    assert set(current) - set(previous) == {1., 3.}  # +2 hours in one stay
    assert set(previous) <= set(current)
    assert current[0.] == previous[0.]  # settings never change
    assert current[2.] == previous[2.]  # the existing measured channel survives
    assert previous[4.] == pytest.approx(9 * factor)
