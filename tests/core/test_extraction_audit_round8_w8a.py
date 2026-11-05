"""Invented source regressions for the round-8 AUMC/HiRID audit."""

import json

import pandas as pd
import pytest

from easyicu.api import concepts as api
from easyicu.api import extraction
from tests.support.aumc_source import aumc_db, aumc_drug
from tests.support.hirid_source import hirid_source_root


@pytest.fixture(autouse=True)
def clear_loader():
    api.clear_global_loader()
    yield
    api.clear_global_loader()


def _load(root, database, names, *, patient_ids=None, **kwargs):
    api.clear_global_loader()
    return api.load_concepts(
        names, database=database, data_path=root,
        patient_ids=[1] if patient_ids is None else patient_ids,
        verbose=False, concept_workers=1, parallel_workers=1, **kwargs,
    )


@pytest.mark.parametrize("database", ["aumc", "hirid"])
@pytest.mark.parametrize("injury_creatinine", [150., 250.])
def test_r8_a1_hyperkalemia_renal_agrees_across_module_single_and_batch(tmp_path, database, injury_creatinine):
    if database == "aumc":
        source = aumc_db(tmp_path / "source", origin=17.25, numeric=[
            (5., 6836, 80., "umol/L"), (5., 6835, 4., "mmol/L"),
            (35., 6835, 4., "mmol/L"),
            (65., 6836, injury_creatinine, "umol/L"), (65., 6835, 4., "mmol/L"),
            (95., 6835, 6.1, "mmol/L"),
        ], layout="bucket")
        root = source.base_path
        identifier = "admissionid"
    else:
        root = hirid_source_root(tmp_path, "observations", [
            {"patientid": 1, "datetime": 5 / 60, "variableid": 20000600, "value": 80.},
            {"patientid": 1, "datetime": 5 / 60, "variableid": 20000500, "value": 4.},
            {"patientid": 1, "datetime": 35 / 60, "variableid": 20000500, "value": 4.},
            {"patientid": 1, "datetime": 65 / 60, "variableid": 20000600, "value": injury_creatinine},
            {"patientid": 1, "datetime": 65 / 60, "variableid": 20000500, "value": 4.},
            {"patientid": 1, "datetime": 95 / 60, "variableid": 20000500, "value": 6.1},
        ])
        identifier = "patientid"
    # A formal module sees the cohort, while the audit's single entry replays
    # one stay. Crossing the pre-sampling threshold must not move its K grid.
    cohort_ids = list(range(1, 1003))
    demographics = root / ("admissions.parquet" if database == "aumc" else "general.parquet")
    stays = pd.read_parquet(demographics).iloc[[0]]
    stays = stays.loc[stays.index.repeat(len(cohort_ids))].reset_index(drop=True)
    stays[identifier] = cohort_ids
    stays.to_parquet(demographics, index=False)
    events = root / ("numericitems_bucket/bucket_id=0/part.parquet" if database == "aumc" else "observations/1.parquet")
    frame = pd.read_parquet(events)
    padding = frame.iloc[[1]]
    padding = padding.loc[padding.index.repeat(len(cohort_ids) - 1)].reset_index(drop=True)
    padding[identifier] = cohort_ids[1:]
    pd.concat([frame, padding], ignore_index=True).to_parquet(events, index=False)

    def check(frame):
        actual = frame.loc[frame[identifier].eq(1)].set_index("charttime").sofa2_renal
        assert actual.loc[0.] == 0
        assert actual.loc[1.] == 4

    check(_load(root, database, ["sofa2"], keep_components=True, r_compatible=False))
    output = tmp_path / "export"
    output.mkdir()
    _load(root, database, ["potassium", "crea"], merge=False, use_sofa2=True)
    extraction._run_module_extraction(
        module_name="sofa2_score", concepts=extraction.EXTRACT_MODULES["sofa2_score"],
        database=database, data_path=str(root), patient_ids_filter={identifier: cohort_ids},
        batch_size=len(cohort_ids), output_dir=str(output), use_sofa2=True,
    )
    manifest = json.loads((output / "_manifest.json").read_text())
    assert manifest["errors"] == []
    check(pd.read_parquet(output / "sofa2_score.parquet"))
    check(_load(
        root, database, ["sofa2", "potassium", "crea", "hr", "k", "na", "lact"],
        patient_ids=cohort_ids, keep_components=True, r_compatible=False,
    ))


@pytest.mark.parametrize("database", ["aumc", "hirid"])
@pytest.mark.parametrize("window", [False, True])
@pytest.mark.parametrize("layout", ["bucket", "file"])
def test_r8_a2_partial_wide_materializes_exact_intervals_before_merge(tmp_path, database, window, layout):
    if database == "aumc":
        source = aumc_db(tmp_path / "source", numeric=[
            (5., 6640, 80., "/min"), (5., 6835, 4., "mmol/L"),
            (5., 6840, 140., "mmol/L"), (5., 6837, 1., "mmol/L"),
        ], drugs=[aumc_drug(17., 137., 7229, dose=.2)], layout=layout)
        root = source.base_path
    else:
        root = hirid_source_root(tmp_path, "pharma", [
            {"patientid": 1, "givenat": 17 / 60, "pharmaid": 1000462,
             "infusionid": 1, "givendose": .2, "route": "iv"},
            {"patientid": 1, "givenat": 137 / 60, "pharmaid": 1000462,
             "infusionid": 1, "givendose": .2, "route": "iv"},
        ])
        if layout == "file":
            (root / "observations/1.parquet").rename(root / "observations.parquet")
            (root / "observations").rmdir()
    names = ["hr", "k", "na", "lact", "norepi_dur", "vaso_ind", "norepi60", "norepi_rate"]
    ledger = _load(root, database, ["norepi_dur"], merge=False, r_compatible=False, interval=None)["norepi_dur"]
    assert ledger.data.norepi_dur.tolist() == pytest.approx([2.])
    assert ledger.data[ledger.index_column].tolist() == pytest.approx([17 / 60])
    native_single = _load(root, database, ["norepi_dur"], r_compatible=False, interval=None)
    assert native_single[ledger.index_column].tolist() == pytest.approx([17 / 60])
    options = {"interval": None, "win_length": "24h"} if window else {}
    merged = _load(root, database, names, merge=True, r_compatible=False, **options)
    assert set(names) <= set(merged.columns)
    if not window:
        output = tmp_path / "vasopressors-export"
        output.mkdir()
        api.clear_global_loader()
        extraction._run_module_extraction(
            module_name="vasopressors", concepts=extraction.EXTRACT_MODULES["vasopressors"],
            database=database, data_path=str(root),
            patient_ids_filter={"admissionid" if database == "aumc" else "patientid": [1]},
            batch_size=1, output_dir=str(output), use_sofa2=True,
        )
        assert json.loads((output / "_manifest.json").read_text())["errors"] == []
        official = pd.read_parquet(output / "vasopressors.parquet").set_index("charttime")
        for name in ("norepi_dur", "vaso_ind", "norepi60", "norepi_rate"):
            pd.testing.assert_series_equal(
                merged.set_index("charttime")[name].dropna().sort_index(),
                official[name].dropna().sort_index(), check_dtype=False, check_index_type=False,
                obj=f"official {name}",
            )
    for name in names:
        single = _load(root, database, [name], merge=True, r_compatible=True, **options)
        actual = merged.set_index("charttime")[name].dropna().sort_index()
        expected = single.set_index("charttime")[name].dropna().sort_index()
        assert not expected.empty, name
        pd.testing.assert_series_equal(
            actual, expected, check_dtype=False, check_names=False, check_index_type=False,
            obj=name,
        )
