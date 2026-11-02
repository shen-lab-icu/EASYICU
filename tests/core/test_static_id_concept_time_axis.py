"""Static DuckDB concepts must keep patient identity separate from event time."""

from types import SimpleNamespace

import pandas as pd
import pytest

from easyicu.concept import ConceptResolver
from easyicu.concept.schema import ConceptDefinition, ConceptDictionary, ConceptSource
from easyicu.datasource import ICUDataSource
from easyicu.resources import load_data_sources
from easyicu.table import has_time_cols


@pytest.mark.parametrize("layout", ["flat", "bucket"])
@pytest.mark.parametrize("selected", [None, [8785, 33905]])
def test_static_duckdb_median_retains_high_patient_ids(tmp_path, layout, selected):
    ids = [1, 8784, 8785, 33905, *range(40000, 40036)]
    rows = pd.DataFrame(
        {
            "patientid": [pid for pid in ids for _ in range(3)],
            "variableid": [10000400] * (3 * len(ids)),
            "value": [value for i in range(len(ids)) for value in [60.0 + i, 80.0 + i, None]],
            "datetime": [pd.Timestamp("2026-01-01")] * (3 * len(ids)),
        }
    )
    directory = tmp_path / (
        "observations" if layout == "flat" else "observations_bucket/bucket_id=0"
    )
    directory.mkdir(parents=True)
    rows.to_parquet(directory / "part.parquet", index=False)
    source = ICUDataSource(load_data_sources().get("hirid"), base_path=tmp_path)
    dictionary = ConceptDictionary(
        {
            "weight": ConceptDefinition(
                name="weight",
                class_name="num_cncpt",
                target="id_tbl",
                sources={
                    "hirid": [
                        ConceptSource(
                            table="observations",
                            sub_var="variableid",
                            ids=[10000400],
                            value_var="value",
                        )
                    ]
                },
            )
        }
    )
    result = ConceptResolver(dictionary).load_concepts(
        ["weight"],
        source,
        merge=False,
        patient_ids=ids if selected is None else selected,
        r_compatible=False,
        concept_workers=1,
        verbose=False,
    )["weight"]
    expected = rows.groupby("patientid", as_index=False)["value"].median()
    expected = expected.rename(columns={"value": "weight"})
    if selected is not None:
        expected = expected[expected.patientid.isin(selected)]
    pd.testing.assert_frame_equal(
        result.data.sort_values("patientid").reset_index(drop=True),
        expected.reset_index(drop=True),
        check_dtype=False,
    )
    assert result.index_column is None
    assert not has_time_cols(result)


def test_actual_hirid_event_time_quarantine_still_applies():
    resolver = ConceptResolver(ConceptDictionary({}))
    source = SimpleNamespace(config=SimpleNamespace(name="hirid"))
    rows = pd.DataFrame(
        {
            "patientid": [33905] * 4,
            "charttime": [-25.0, 0.0, 8784.0, 8785.0],
            "map": [80.0] * 4,
        }
    )
    result = resolver._align_time_to_admission(
        rows, source, ["patientid"], "charttime"
    )
    assert result.patientid.tolist() == [33905, 33905]
    assert result.charttime.tolist() == [0.0, 8784.0]
