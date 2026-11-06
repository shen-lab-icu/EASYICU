"""A HiRID stay's values must not depend on which other stays share its batch.

A former loader step re-bucketed a loaded HiRID/AUMC frame to hourly medians
whenever it held more than 1,000 rows and the median gap across all patients'
rows was under half the interval.  A stay loaded alone kept its raw rows; the
same stay loaded with a full batch was reduced to hourly medians first.  The
weight read inside ``hirid_rate_kg`` therefore moved with the batch, and with
it every per-kg drug rate (p9b HiRID sample: propofol_rate 63 cells,
norepi_rate 9 cells between the default and the 1,000-stay batch).  ricu
bounds each value and aggregates once, so the raw median is the reference.
"""

import pandas as pd
import pytest

from easyicu.concept import ConceptResolver
from easyicu.concept.schema import ConceptDefinition, ConceptDictionary, ConceptSource
from easyicu.datasource import ICUDataSource
from easyicu.resources import load_data_sources

TARGET = 7
# Hour 0 holds three readings, hours 1 and 2 one each: the raw median is 74,
# the median of the hourly medians (72, 90, 91) would be 90.
TARGET_ROWS = [(0.0, 70.0), (0.2, 72.0), (0.4, 74.0), (1.0, 90.0), (2.0, 91.0)]
BACKGROUND = list(range(1000, 1060))  # 60 stays x 20 rows -> 1,200 rows


def _source(tmp_path):
    base = pd.Timestamp("2026-01-01")
    rows = [
        {"patientid": TARGET, "variableid": 10000400, "value": value,
         "datetime": base + pd.Timedelta(hours=hours)}
        for hours, value in TARGET_ROWS
    ]
    rows += [
        {"patientid": pid, "variableid": 10000400, "value": 60.0 + (k % 7),
         "datetime": base + pd.Timedelta(minutes=2 * k)}
        for pid in BACKGROUND for k in range(20)
    ]
    directory = tmp_path / "observations"
    directory.mkdir()
    pd.DataFrame(rows).to_parquet(directory / "part.parquet", index=False)
    pd.DataFrame(
        {"patientid": [TARGET, *BACKGROUND], "admissiontime": base}
    ).to_parquet(tmp_path / "general_table.parquet", index=False)
    return ICUDataSource(load_data_sources().get("hirid"), base_path=tmp_path)


def _dictionary():
    return ConceptDictionary({
        "weight": ConceptDefinition(
            name="weight", class_name="num_cncpt", target="id_tbl",
            sources={"hirid": [ConceptSource(
                table="observations", sub_var="variableid", ids=[10000400], value_var="value",
            )]},
        )
    })


def _target_weight(source, patient_ids, **kwargs):
    result = ConceptResolver(_dictionary()).load_concepts(
        ["weight"], source, merge=False, patient_ids=patient_ids,
        r_compatible=False, concept_workers=1, verbose=False, **kwargs,
    )["weight"]
    data = result.data
    return data.loc[data.patientid.eq(TARGET), "weight"].tolist()


@pytest.mark.parametrize("cohort", [[TARGET], [TARGET, *BACKGROUND[:5]], [TARGET, *BACKGROUND]])
def test_static_weight_is_the_raw_median_in_any_batch(tmp_path, cohort):
    assert _target_weight(_source(tmp_path), cohort) == [74.0]


def test_raw_rows_do_not_depend_on_the_batch(tmp_path):
    """The callback-internal weight load (aggregator=False) keeps raw rows."""
    source = _source(tmp_path)
    resolver = ConceptResolver(_dictionary())

    def raw(ids):
        table = resolver._load_single_concept(
            "weight", source, aggregator=False,
            patient_ids={"patientid": ids}, verbose=False, _bypass_callback=True,
        )
        data = table.data
        return sorted(data.loc[data.patientid.eq(TARGET), "weight"].tolist())

    expected = sorted(value for _, value in TARGET_ROWS)
    assert raw([TARGET]) == expected
    assert raw([TARGET, *BACKGROUND]) == expected
