"""Exercise the independent SQL oracle with synthetic source records."""
import json
import runpy
from pathlib import Path
from types import SimpleNamespace

import duckdb
import pandas as pd
import pytest


@pytest.mark.parametrize('fault', [None, 'float32', 'hybrid', 'tidal', 'missing'])
def test_raw_oracle_validates_values_and_source_completeness(tmp_path, fault):
    verify = runpy.run_path(str(Path(__file__).resolve().parents[2] /
                               'tools/verify_aumc_ventilator_sources.py'))['verify']
    raw = tmp_path / 'raw'
    (raw / 'listitems').mkdir(parents=True)
    (raw / 'numericitems').mkdir()
    origin = 5400000.
    pd.DataFrame({'admissionid':[1], 'admittedat':[origin],
                  'dischargedat':[origin+3600000.]}).to_parquet(raw / 'admissions.parquet')
    pd.DataFrame({'admissionid':[1,1], 'measuredat':[origin+60000.]*2,
                  'itemid':[12290,12347], 'value':['CPPV','SIMV_ASB']}).to_parquet(raw / 'listitems/1.parquet')
    tidal_values = [500.123456, 501.987654] if fault == 'float32' else [0.,2849.]
    pd.DataFrame({'admissionid':[1,1], 'measuredat':[origin+60000.,origin+120000.],
                  'itemid':[12275,12277], 'value':tidal_values,
                  'unit':['ml','ml']}).to_parquet(raw / 'numericitems/1.parquet')
    before = pd.DataFrame({'stay_id':[1], 'charttime':[0.],
                           'vent_mode':['unspecified'], 'vent_breath_seq':['controlled'],
                           'tidal_vol':[1424.5]})
    after = before.copy()
    after['vent_mode'] = 'volume'
    after['tidal_vol'] = 0.
    if fault == 'float32':
        after['tidal_vol'] = pd.Series([sum(tidal_values)/2], dtype='float32').item()
    if fault == 'hybrid':
        after['vent_mode'] = 'unspecified'
    elif fault == 'tidal':
        after['tidal_vol'] = 1424.5
    elif fault == 'missing':
        after = after.iloc[:0]
    before.to_parquet(tmp_path / 'before.parquet', index=False)
    after.to_parquet(tmp_path / 'after.parquet', index=False)
    args = SimpleNamespace(raw=str(raw), before=str(tmp_path / 'before.parquet'),
                           after=str(tmp_path / 'after.parquet'))
    output = tmp_path / 'receipt.json'
    with duckdb.connect() as con:
        con.execute('SET threads=1')
        if fault in (None, 'float32'):
            verify(args, output, con)
        else:
            with pytest.raises(SystemExit, match='1'):
                verify(args, output, con)
    result = json.loads(output.read_text())
    assert result['passed'] is (fault in (None, 'float32'))
    if fault == 'float32':
        assert result['tidal_source_mismatches'] == 1
        assert result['tidal_float32_source_mismatches'] == 0
    elif fault == 'hybrid':
        assert result['mode_source_mismatches'] == 1
    elif fault == 'tidal':
        assert result['tidal_source_mismatches'] == 1
    elif fault == 'missing':
        assert result['missing_mode_outputs'] == result['missing_tidal_outputs'] == 1
