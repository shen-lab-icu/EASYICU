"""An all-ICU export's cohort report names the stay table's format (delta17c).

The CSV sibling stands in only when the Parquet stay table is absent, and
the published cohort report says so (``source_format: csv_fallback``).
Rows are synthetic.
"""

from __future__ import annotations

import pandas as pd

from easyicu import api
from easyicu.webserver import dataio


def test_an_absent_parquet_stay_table_is_reported_as_csv_fallback(tmp_path):
    pd.DataFrame({"icustay_id": [101, 102]}).to_csv(tmp_path / "icustays.csv", index=False)
    result = dataio._resolve_export_cohort(str(tmp_path), "mimic", {"preset": "all_icu"}, None, api)
    assert result["cohort_report"]["source_format"] == "csv_fallback"


def test_a_parquet_stay_table_is_reported_as_parquet(tmp_path):
    pd.DataFrame({"icustay_id": [101, 102]}).to_parquet(tmp_path / "icustays.parquet", index=False)
    result = dataio._resolve_export_cohort(str(tmp_path), "mimic", {"preset": "all_icu"}, None, api)
    assert result["cohort_report"]["source_format"] == "parquet"
