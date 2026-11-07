"""HiRID observation reads follow the caller's source configuration (r13srccfg).

The death callbacks and the native ICU-death timing read HiRID observations
through ``hirid_observation_read_exprs``.  It used to re-load the packaged
HiRID configuration by name and pair it with the caller's root, so a custom
or attached configuration with another observation mapping read the packaged
mapping's files (here: none) without failing.
"""

from __future__ import annotations

import pandas as pd
import pytest

from easyicu.concept.callback_apply import hirid_observation_read_exprs
from easyicu.config import TableConfig
from easyicu.datasource import ICUDataSource
from easyicu.io.data_tools import load_src_cfg
from easyicu.runtime import duckdb_spill


def _custom_root(tmp_path):
    """Synthetic observations under a site-specific mapping (two parquet parts)."""
    site = tmp_path / "obs_site"
    site.mkdir()
    for part, (pid, hours) in enumerate(((1, 2), (1, 5))):
        pd.DataFrame({
            "patientid": [pid], "variableid": [200], "value": [80.0],
            "datetime": [pd.Timestamp("2100-01-01") + pd.Timedelta(hours=hours)],
        }).to_parquet(site / f"p{part}.parquet")
    config = load_src_cfg("hirid").model_copy(deep=True)
    config.tables["observations"] = TableConfig(
        name="observations",
        files=[{"path": "obs_site/p0.parquet"}, {"path": "obs_site/p1.parquet"}],
        defaults=config.tables["observations"].defaults,
    )
    return ICUDataSource(config, base_path=tmp_path)


def test_observation_reads_follow_a_custom_mapping(tmp_path):
    source = _custom_root(tmp_path)
    expressions = hirid_observation_read_exprs(source)
    assert expressions is not None
    with duckdb_spill.connect() as conn:
        rows = conn.execute(f"SELECT count(*) FROM {expressions[1]}").fetchone()[0]
    assert rows == 2


def test_the_packaged_mapping_under_that_root_finds_nothing(tmp_path):
    # The mismatch the helper used to make silently.
    _custom_root(tmp_path)
    packaged = ICUDataSource(load_src_cfg("hirid"), base_path=tmp_path)
    assert hirid_observation_read_exprs(packaged) is None


def test_icu_death_timing_reads_the_callers_source(tmp_path):
    from easyicu.scores.icu_mortality import _hirid_time

    source = _custom_root(tmp_path)
    frame = pd.DataFrame({"patientid": [1], "admissiontime": [pd.Timestamp("2100-01-01")]})
    hours = _hirid_time(tmp_path, frame, source)
    assert hours.tolist() == [5.0]


def test_the_helper_refuses_a_bare_root(tmp_path):
    with pytest.raises(TypeError, match="ICUDataSource"):
        hirid_observation_read_exprs(tmp_path)
